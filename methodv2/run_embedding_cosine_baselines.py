#!/usr/bin/env python3
"""Embedding-only cosine baselines for hidden-column reasoning questions.

This runner mirrors the hidden-column setup used by the hybrid LLM pipeline, but
replaces the reasoning step with nearest-neighbor retrieval over embeddings.

For each eligible question:
1. Hide the target column from a copied SQLite database.
2. Execute the ground-truth SQL to obtain visible evidence rows.
3. For each visible row, predict the hidden value using cosine-similarity
   retrieval under three views:
   - column: source column value only
   - row: all visible evidence columns
   - tuple: question + visible evidence row
4. Save row-level predictions, question-level summaries, and per-question CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from question_table_exports import export_question_tables  # noqa: E402
from run_hidden_column_sql_eval import (  # noqa: E402
    DEFAULT_DB_PATH,
    create_reduced_db,
    fetch_schema,
    load_csv_rows,
    make_run_dir,
    sanitize_name,
)
from utils import append_jsonl, quote_ident, setup_logger, write_json, write_jsonl  # noqa: E402

DEFAULT_CSV_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches_table_rows.csv"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"
DEFAULT_BERT_MODEL = (
    "/mnt/shared/shared_hf_home/hub/models--bert-base-uncased/snapshots/"
    "86b5e0934494bd15c9632b12f734a8a67f723594"
)
DEFAULT_SBERT_MODEL = (
    "/mnt/shared/shared_hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
DEFAULT_BGE_MODEL = (
    "/mnt/shared/shared_hf_home/hub/models--BAAI--bge-base-en-v1.5/"
    "snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run embedding-only cosine baselines for the strict table split. "
            "The target column is hidden from the evidence rows, and nearest-neighbor "
            "retrieval is used to predict the hidden value."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--sql_key", default="sql_query")
    ap.add_argument("--column_used_key", default="column_used")
    ap.add_argument("--target_column_key", default="ground_truth_column")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--csv_row_number", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_prompt_rows", type=int, default=50)
    ap.add_argument("--max_candidate_rows_per_target", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--checkpoint_every", type=int, default=1)
    ap.add_argument("--bert_model_path", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--sbert_model_path", default=DEFAULT_SBERT_MODEL)
    ap.add_argument("--bge_model_path", default=DEFAULT_BGE_MODEL)
    ap.add_argument(
        "--baseline_models",
        nargs="+",
        default=["bert", "sbert", "bge"],
        choices=["bert", "sbert", "bge"],
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def canonical_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    if value is None:
        return None
    return str(value).strip()


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return json.dumps(canonical_value(value), ensure_ascii=False)


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def parse_target_columns(text: str) -> List[str]:
    target = (text or "").strip()
    if not target or target == "no_match" or " -> " in target or " | " in target:
        return []
    return [target]


def build_select_variant(sql: str, select_exprs: Sequence[str]) -> str:
    sql0 = (sql or "").strip().rstrip(";")
    import re

    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    return f"SELECT {', '.join(select_exprs)} {sql0[match.start():]};"


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.execute((sql or "").strip().rstrip(";"))
    cols = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall() if cur.description else []
    return cols, rows


def sql_references_column(sql: str, column_name: str) -> bool:
    sql_text = (sql or "").strip()
    if not sql_text or not column_name:
        return False
    quoted = quote_ident(column_name)
    if quoted in sql_text:
        return True
    return column_name.lower() in sql_text.lower()


def rows_to_objects(cols: Sequence[str], rows: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        obj = {}
        for idx, col in enumerate(cols):
            obj[col] = canonical_value(row[idx] if idx < len(row) else None)
        out.append(obj)
    return out


def row_key(row_obj: Dict[str, Any]) -> str:
    return json.dumps({k: canonical_value(v) for k, v in row_obj.items()}, ensure_ascii=False, sort_keys=True)


def dedupe_preserve(values: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for value in values:
        key = json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical_value(value))
    return out


def build_actual_mapping(visible_cols: Sequence[str], rows_with_hidden: Sequence[Tuple[Any, ...]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    visible_count = len(visible_cols)
    for row in rows_with_hidden:
        visible_obj = {
            visible_cols[idx]: canonical_value(row[idx] if idx < len(row) else None)
            for idx in range(visible_count)
        }
        hidden_value = canonical_value(row[visible_count] if len(row) > visible_count else None)
        key = row_key(visible_obj)
        entry = mapping.setdefault(
            key,
            {
                "visible_row": visible_obj,
                "actual_values": [],
            },
        )
        entry["actual_values"].append(hidden_value)
    for entry in mapping.values():
        entry["actual_values"] = dedupe_preserve(entry["actual_values"])
    return mapping


def preferred_evidence_columns(schema_cols: Sequence[str], selected_cols: Sequence[str], hidden_column: str) -> List[str]:
    preferred = [col for col in ("NCT", "PubMed ID", "Trial name") if col in set(schema_cols)]
    out: List[str] = []
    seen = set()
    for col in preferred + list(selected_cols):
        if not col or col == hidden_column or col in seen:
            continue
        if col not in schema_cols:
            continue
        seen.add(col)
        out.append(col)
    return out


def compute_similarity(predicted: Any, actual_options: Sequence[Any]) -> float:
    pred = canonical_value(predicted)
    actuals = [canonical_value(x) for x in actual_options]
    if pred in actuals:
        return 1.0
    pred_text = normalize_text(pred)
    if not pred_text:
        return 0.0
    for option in actuals:
        if pred_text == normalize_text(option):
            return 1.0
    return 0.0


def build_column_text(value: Any) -> str:
    return f"value: {format_value(value)}"


def build_row_text(row_obj: Dict[str, Any]) -> str:
    parts = [f"{key}: {format_value(value)}" for key, value in sorted(row_obj.items())]
    return " | ".join(parts)


def build_tuple_text(question: str, row_obj: Dict[str, Any]) -> str:
    return f"question: {question}\nrow: {build_row_text(row_obj)}"


@dataclass
class CandidateRow:
    row_id: int
    visible_row: Dict[str, Any]
    source_value: Any
    target_value: Any


class LocalEmbedder:
    def __init__(self, model_path: str, *, device: str, instruction_prefix: str = "") -> None:
        self.model_path = model_path
        self.device = device
        self.instruction_prefix = instruction_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        self.model.to(device)

    def _prepare_text(self, text: str) -> str:
        if self.instruction_prefix:
            return f"{self.instruction_prefix}{text}"
        return text

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            hidden = getattr(self.model.config, "hidden_size", 768)
            return np.zeros((0, hidden), dtype=np.float32)
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [self._prepare_text(text) for text in texts[start : start + batch_size]]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=min(getattr(self.tokenizer, "model_max_length", 512), 512),
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                model_out = self.model(**batch)
                token_embeddings = model_out.last_hidden_state
                attention_mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts
                normed = F.normalize(mean_pooled, p=2, dim=1)
                outputs.append(normed.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)


def cosine_top1(query_vec: np.ndarray, matrix: np.ndarray) -> Tuple[int, float]:
    scores = np.matmul(matrix, query_vec)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return best_idx, best_score


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return (sum(values) / len(values)) if values else 0.0


def write_csv(path: Path, rows_out: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    for row in rows_out:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def choose_column_text(row_obj: Dict[str, Any], source_column: str) -> str:
    return build_column_text(row_obj.get(source_column))


def build_candidate_rows(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    source_column: str,
    target_column: str,
    visible_columns: Sequence[str],
    max_rows: int,
) -> List[CandidateRow]:
    select_cols: List[str] = []
    seen = set()
    for col in list(visible_columns) + [source_column, target_column]:
        if col and col not in seen:
            seen.add(col)
            select_cols.append(col)
    sql = f"SELECT {', '.join(quote_ident(col) for col in select_cols)} FROM {quote_ident(table_name)}"
    rows_cols, rows = execute_query(conn, sql)
    out: List[CandidateRow] = []
    for row_id, row in enumerate(rows, start=1):
        obj = {rows_cols[idx]: canonical_value(row[idx]) for idx in range(len(rows_cols))}
        target_value = obj.get(target_column)
        if target_value is None or str(target_value).strip() == "":
            continue
        visible_obj = {col: obj.get(col) for col in visible_columns if col in obj}
        out.append(
            CandidateRow(
                row_id=row_id,
                visible_row=visible_obj,
                source_value=obj.get(source_column),
                target_value=target_value,
            )
        )
        if max_rows and len(out) >= max_rows:
            break
    return out


def load_questions(rows: Sequence[Dict[str, str]], args: argparse.Namespace, schema_cols: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    usable_schema = set(schema_cols)
    items: List[Dict[str, Any]] = []
    skipped = Counter()
    for csv_row_number, row in enumerate(rows, start=2):
        if args.csv_row_number and csv_row_number != args.csv_row_number:
            continue
        question = (row.get(args.question_key) or "").strip()
        gt_sql = (row.get(args.sql_key) or "").strip()
        source_column = (row.get(args.column_used_key) or "").strip()
        target_columns = parse_target_columns((row.get(args.target_column_key) or "").strip())
        if not question:
            skipped["missing_question"] += 1
            continue
        if not gt_sql:
            skipped["missing_gt_sql"] += 1
            continue
        if source_column not in usable_schema:
            skipped["source_not_usable"] += 1
            continue
        if len(target_columns) != 1 or target_columns[0] not in usable_schema:
            skipped["target_not_usable"] += 1
            continue
        items.append(
            {
                "item_id": f"row_{csv_row_number}",
                "csv_row_number": csv_row_number,
                "question": question,
                "gt_sql": gt_sql,
                "column_used": source_column,
                "target_column": target_columns[0],
            }
        )
        if args.limit and len(items) >= args.limit:
            break
    return items, dict(skipped)


def model_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    specs = []
    for name in args.baseline_models:
        if name == "bert":
            specs.append(("bert", args.bert_model_path, ""))
        elif name == "sbert":
            specs.append(("sbert", args.sbert_model_path, ""))
        elif name == "bge":
            specs.append(("bge", args.bge_model_path, "Represent this sentence for retrieval: "))
    return specs


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_name = args.run_name.strip() or "embedding_cosine_baselines"
    run_dir = make_run_dir(run_root, run_name)
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"embed_baseline_{run_dir.name}")

    logger.info("CSV path: %s", csv_path)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Dry run: %s", bool(args.dry_run))

    rows = load_csv_rows(csv_path)
    original_conn = sqlite3.connect(db_path)
    try:
        schema_cols = fetch_schema(original_conn, args.table_name)
    finally:
        original_conn.close()

    items, skipped = load_questions(rows, args, schema_cols)
    groups = Counter(item["target_column"] for item in items)
    logger.info("Eligible questions: %d", len(items))
    logger.info("Skipped: %s", skipped)
    logger.info("Groups: %s", dict(groups))

    model_specs = model_specs_from_args(args)
    meta = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "question_count": len(items),
        "skipped": skipped,
        "models": [{"name": name, "path": path} for name, path, _ in model_specs],
    }
    write_json(run_dir / "run_meta.json", meta)

    completed_item_ids: set[str] = set()
    question_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []
    question_checkpoint_jsonl = run_dir / "question_results_checkpoint.jsonl"
    row_checkpoint_jsonl = run_dir / "row_level_checkpoint.jsonl"
    if question_checkpoint_jsonl.exists():
        with question_checkpoint_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                question_rows.append(obj)
                completed_item_ids.add(str(obj.get("item_id")))
    if row_checkpoint_jsonl.exists():
        with row_checkpoint_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row_rows.append(json.loads(line))

    embedders: Dict[str, LocalEmbedder] = {}
    if not args.dry_run:
        for name, model_path, instruction_prefix in model_specs:
            logger.info("Loading embedder %s from %s", name, model_path)
            embedders[name] = LocalEmbedder(model_path, device=args.device, instruction_prefix=instruction_prefix)

    original_conn = sqlite3.connect(db_path)
    questions_root_dir = run_dir / "questions"
    try:
        for item in items:
            if item["item_id"] in completed_item_ids:
                logger.info("Skipping completed item %s", item["item_id"])
                continue

            target_column = item["target_column"]
            question = item["question"]
            source_column = item["column_used"]
            logger.info("Processing %s target=%s", item["item_id"], target_column)

            group_dir = run_dir / f"group__{sanitize_name(target_column)}"
            group_dir.mkdir(parents=True, exist_ok=True)
            reduced_db_path = group_dir / f"{item['item_id']}__reduced.db"
            create_reduced_db(
                source_db=db_path,
                output_db=reduced_db_path,
                table_name=args.table_name,
                removed_column=target_column,
            )
            reduced_conn = sqlite3.connect(reduced_db_path)
            try:
                reduced_schema = fetch_schema(reduced_conn, args.table_name)
                gt_sql_uses_hidden = sql_references_column(item["gt_sql"], target_column)
                gt_query_conn = original_conn if gt_sql_uses_hidden else reduced_conn
                gt_query_schema = list(schema_cols) if gt_sql_uses_hidden else list(reduced_schema)
                gt_visible_source_db = "original" if gt_sql_uses_hidden else "reduced"
                gt_visible_cols: List[str] = []
                gt_visible_rows: List[Tuple[Any, ...]] = []
                gt_visible_error = ""
                gt_evidence_sql = ""
                gt_evidence_cols: List[str] = []
                gt_evidence_rows: List[Tuple[Any, ...]] = []
                gt_evidence_error = ""
                gt_actual_mapping: Dict[str, Dict[str, Any]] = {}
                gt_actual_sql = ""
                gt_actual_error = ""
                try:
                    gt_visible_cols, gt_visible_rows = execute_query(gt_query_conn, item["gt_sql"])
                except Exception as exc:
                    gt_visible_error = str(exc)
                if gt_visible_cols:
                    try:
                        preferred_cols = preferred_evidence_columns(gt_query_schema, gt_visible_cols, target_column)
                        if not preferred_cols:
                            raise ValueError("no visible evidence columns remain after hiding target column")
                        gt_evidence_sql = build_select_variant(
                            item["gt_sql"],
                            [quote_ident(col) for col in preferred_cols],
                        )
                        gt_evidence_cols, gt_evidence_rows = execute_query(gt_query_conn, gt_evidence_sql)
                        gt_actual_sql = build_select_variant(
                            gt_evidence_sql,
                            [quote_ident(col) for col in gt_evidence_cols] + [quote_ident(target_column)],
                        )
                        _, gt_actual_rows = execute_query(original_conn, gt_actual_sql)
                        gt_actual_mapping = build_actual_mapping(gt_evidence_cols, gt_actual_rows)
                    except Exception as exc:
                        if not gt_evidence_error:
                            gt_evidence_error = str(exc)
                        gt_actual_error = str(exc)

                prompted_row_objects = rows_to_objects(gt_evidence_cols, gt_evidence_rows)
                if args.max_prompt_rows and len(prompted_row_objects) > args.max_prompt_rows:
                    prompted_row_objects = prompted_row_objects[: args.max_prompt_rows]

                model_question_metrics: Dict[str, Dict[str, Any]] = {}
                best_pred_by_row_index: Dict[int, Any] = {}
                best_model_name = ""
                best_view_name = ""
                best_metric_key = (-1.0, -1.0, -1.0)
                per_question_prediction_rows: List[Dict[str, Any]] = []

                if not args.dry_run and prompted_row_objects:
                    candidate_rows = build_candidate_rows(
                        original_conn,
                        table_name=args.table_name,
                        source_column=source_column,
                        target_column=target_column,
                        visible_columns=gt_evidence_cols,
                        max_rows=args.max_candidate_rows_per_target,
                    )
                    if candidate_rows:
                        column_candidate_texts = [build_column_text(candidate.source_value) for candidate in candidate_rows]
                        row_candidate_texts = [build_row_text(candidate.visible_row) for candidate in candidate_rows]
                        tuple_candidate_texts = [build_tuple_text(question, candidate.visible_row) for candidate in candidate_rows]

                        for model_name, embedder in embedders.items():
                            column_candidate_matrix = embedder.encode(column_candidate_texts, batch_size=args.batch_size)
                            row_candidate_matrix = embedder.encode(row_candidate_texts, batch_size=args.batch_size)
                            tuple_candidate_matrix = embedder.encode(tuple_candidate_texts, batch_size=args.batch_size)

                            per_view_predictions: Dict[str, Dict[int, Any]] = {}
                            per_view_scores: Dict[str, Dict[int, float]] = {}
                            for view_name in ("column", "row", "tuple"):
                                if view_name == "column":
                                    query_texts = [choose_column_text(row_obj, source_column) for row_obj in prompted_row_objects]
                                    candidate_matrix = column_candidate_matrix
                                elif view_name == "row":
                                    query_texts = [build_row_text(row_obj) for row_obj in prompted_row_objects]
                                    candidate_matrix = row_candidate_matrix
                                else:
                                    query_texts = [build_tuple_text(question, row_obj) for row_obj in prompted_row_objects]
                                    candidate_matrix = tuple_candidate_matrix

                                query_matrix = embedder.encode(query_texts, batch_size=args.batch_size)
                                pred_by_row_index: Dict[int, Any] = {}
                                score_by_row_index: Dict[int, float] = {}
                                matched_rows = 0
                                total_rows = 0
                                similarity_sum = 0.0
                                cosine_sum = 0.0
                                for row_index, query_vec in enumerate(query_matrix, start=1):
                                    best_idx, best_score = cosine_top1(query_vec, candidate_matrix)
                                    pred_by_row_index[row_index] = candidate_rows[best_idx].target_value
                                    score_by_row_index[row_index] = best_score
                                    row_obj = prompted_row_objects[row_index - 1]
                                    actual_entry = gt_actual_mapping.get(row_key(row_obj), {"actual_values": []})
                                    actual_values = actual_entry.get("actual_values") or []
                                    predicted_value = pred_by_row_index.get(row_index)
                                    similarity = compute_similarity(predicted_value, actual_values)
                                    matched_rows += int(similarity >= 1.0 - 1e-12)
                                    similarity_sum += similarity
                                    total_rows += 1
                                    cosine_sum += float(best_score)
                                    row_record = {
                                        "item_id": item["item_id"],
                                        "csv_row_number": item["csv_row_number"],
                                        "question": question,
                                        "column_used": source_column,
                                        "final_column": target_column,
                                        "model_name": model_name,
                                        "view_name": view_name,
                                        "row_index": row_index,
                                        "visible_row_json": json.dumps(row_obj, ensure_ascii=False, sort_keys=True),
                                        "predicted_hidden_value": format_value(predicted_value),
                                        "actual_hidden_values_json": json.dumps(actual_values, ensure_ascii=False),
                                        "exact_match": int(similarity >= 1.0 - 1e-12),
                                        "similarity": similarity,
                                        "cosine_score": float(best_score),
                                    }
                                    row_rows.append(row_record)
                                    per_question_prediction_rows.append(row_record)
                                    if args.checkpoint_every and (len(row_rows) % args.checkpoint_every == 0):
                                        append_jsonl(str(row_checkpoint_jsonl), row_record)
                                per_view_predictions[view_name] = pred_by_row_index
                                per_view_scores[view_name] = score_by_row_index
                                exact_match_rate = (matched_rows / total_rows) if total_rows else 0.0
                                avg_similarity = (similarity_sum / total_rows) if total_rows else 0.0
                                avg_cosine = (cosine_sum / total_rows) if total_rows else 0.0
                                model_question_metrics[f"{model_name}:{view_name}"] = {
                                    "exact_match_rate": exact_match_rate,
                                    "avg_similarity": avg_similarity,
                                    "avg_cosine": avg_cosine,
                                    "pred_by_row_index": pred_by_row_index,
                                }
                                score_key = (exact_match_rate, avg_similarity, avg_cosine)
                                if score_key > best_metric_key:
                                    best_metric_key = score_key
                                    best_model_name = model_name
                                    best_view_name = view_name
                                    best_pred_by_row_index = dict(pred_by_row_index)

                question_export_paths = export_question_tables(
                    questions_root_dir=questions_root_dir,
                    item_id=item["item_id"],
                    question=question,
                    hidden_column=target_column,
                    group_name=sanitize_name(target_column),
                    prompt_mode="embedding_baseline",
                    gt_sql=item["gt_sql"],
                    pred_sql="",
                    gt_visible_cols=gt_evidence_cols,
                    gt_visible_rows=rows_to_objects(gt_evidence_cols, gt_evidence_rows),
                    pred_visible_cols=gt_evidence_cols,
                    pred_visible_rows=prompted_row_objects,
                    pred_by_row_index=best_pred_by_row_index,
                    actual_mapping=gt_actual_mapping,
                    row_key_fn=row_key,
                )
                write_csv(Path(question_export_paths["question_dir"]) / "embedding_predictions.csv", per_question_prediction_rows)

                question_row = {
                    "item_id": item["item_id"],
                    "csv_row_number": item["csv_row_number"],
                    "question": question,
                    "column_used": source_column,
                    "final_column": target_column,
                    "gt_sql": item["gt_sql"],
                    "gt_visible_source_db": gt_visible_source_db,
                    "gt_visible_columns_json": json.dumps(gt_visible_cols, ensure_ascii=False),
                    "gt_visible_rows_json": json.dumps(rows_to_objects(gt_visible_cols, gt_visible_rows), ensure_ascii=False),
                    "gt_visible_exec_error": gt_visible_error,
                    "gt_evidence_sql": gt_evidence_sql,
                    "gt_evidence_columns_json": json.dumps(gt_evidence_cols, ensure_ascii=False),
                    "gt_evidence_rows_json": json.dumps(rows_to_objects(gt_evidence_cols, gt_evidence_rows), ensure_ascii=False),
                    "gt_evidence_exec_error": gt_evidence_error,
                    "gt_actual_sql": gt_actual_sql,
                    "gt_actual_mapping_json": json.dumps(gt_actual_mapping, ensure_ascii=False),
                    "gt_actual_exec_error": gt_actual_error,
                    "prompted_row_count": len(prompted_row_objects),
                    "best_model_name": best_model_name,
                    "best_view_name": best_view_name,
                    "metrics_json": json.dumps(model_question_metrics, ensure_ascii=False),
                    "question_dir": question_export_paths["question_dir"],
                    "final_table_csv": question_export_paths["final_table_csv"],
                    "ground_truth_table_csv": question_export_paths["ground_truth_table_csv"],
                }
                question_rows.append(question_row)
                append_jsonl(str(question_checkpoint_jsonl), question_row)
                completed_item_ids.add(item["item_id"])
            finally:
                reduced_conn.close()
    finally:
        original_conn.close()

    question_csv = run_dir / "all_question_results.csv"
    row_csv = run_dir / "row_level_predictions.csv"
    write_csv(question_csv, question_rows)
    write_csv(row_csv, row_rows)
    write_jsonl(question_checkpoint_jsonl, question_rows)
    write_jsonl(row_checkpoint_jsonl, row_rows)

    summary_rows: List[Dict[str, Any]] = []
    by_model_view: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in row_rows:
        key = (str(row.get("model_name") or ""), str(row.get("view_name") or ""))
        by_model_view.setdefault(key, []).append(row)
    for (model_name, view_name), rows_out in sorted(by_model_view.items()):
        summary_rows.append(
            {
                "model_name": model_name,
                "view_name": view_name,
                "row_count": len(rows_out),
                "avg_exact_match": avg_numeric(rows_out, "exact_match"),
                "avg_similarity": avg_numeric(rows_out, "similarity"),
                "avg_cosine_score": avg_numeric(rows_out, "cosine_score"),
            }
        )
    summary_csv = run_dir / "baseline_summary.csv"
    write_csv(summary_csv, summary_rows)

    overall = {
        "meta": meta,
        "question_count": len(question_rows),
        "row_prediction_count": len(row_rows),
        "outputs": {
            "question_results_csv": str(question_csv),
            "row_predictions_csv": str(row_csv),
            "baseline_summary_csv": str(summary_csv),
        },
    }
    write_json(run_dir / "summary.json", overall)
    logger.info("Wrote question results CSV: %s", question_csv)
    logger.info("Wrote row-level CSV: %s", row_csv)
    logger.info("Wrote baseline summary CSV: %s", summary_csv)


if __name__ == "__main__":
    main()
