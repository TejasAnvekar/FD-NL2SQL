#!/usr/bin/env python3
"""Strict cosine baselines using only a copied table plus the question.

This runner samples question-level table-derivation exports, but does not use
their ground-truth rows as retrieval candidates. Instead, for each question it:

1. Loads the question and column_used from the exported metadata.
2. Creates a per-question copy of the full `clinical_trials` table as CSV.
3. Uses the question text as the query and scores candidate rows from the table
   with embedding cosine similarity under three views:
   - column: the row's source-column value only
   - row: the full row text
   - tuple: question + full row text
4. Scores retrieval against the saved ground_truth_table.csv only after ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_embedding_cosine_baselines import (  # noqa: E402
    DEFAULT_BERT_MODEL,
    DEFAULT_BGE_MODEL,
    DEFAULT_SBERT_MODEL,
    LocalEmbedder,
    avg_numeric,
    build_row_text,
    build_tuple_text,
    write_csv,
)
from run_hidden_column_sql_eval import make_run_dir, sanitize_name  # noqa: E402
from utils import setup_logger, write_json  # noqa: E402

DEFAULT_MANIFEST = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv"
DEFAULT_DB_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run strict cosine baselines where each question only has the raw "
            "clinical_trials table copy plus the question text."
        )
    )
    ap.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="question_table_copy_cosine_100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--baseline_models",
        nargs="+",
        default=["bert", "sbert", "bge"],
        choices=["bert", "sbert", "bge"],
    )
    ap.add_argument("--bert_model_path", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--sbert_model_path", default=DEFAULT_SBERT_MODEL)
    ap.add_argument("--bge_model_path", default=DEFAULT_BGE_MODEL)
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def canonical_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [canonical_jsonable(v) for v in value]
    return canonical_value(value)


def canonical_payload_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
                return json.dumps(canonical_jsonable(parsed), ensure_ascii=False, sort_keys=True)
            except Exception:
                pass
    return json.dumps(canonical_jsonable(value), ensure_ascii=False, sort_keys=True)


def exact_match(predicted_payload: Any, actual_payload: Any) -> float:
    return 1.0 if canonical_payload_text(predicted_payload) == canonical_payload_text(actual_payload) else 0.0


def model_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for name in args.baseline_models:
        if name == "bert":
            specs.append(("bert", args.bert_model_path, ""))
        elif name == "sbert":
            specs.append(("sbert", args.sbert_model_path, ""))
        elif name == "bge":
            specs.append(("bge", args.bge_model_path, "Represent this sentence for retrieval: "))
    return specs


def fetch_full_table(conn: sqlite3.Connection, table_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    cur = conn.execute(f'SELECT rowid AS "__rowid__", * FROM "{table_name}"')
    cols = [desc[0] for desc in cur.description]
    rows = []
    for raw in cur.fetchall():
        rows.append({cols[idx]: canonical_value(raw[idx]) for idx in range(len(cols))})
    return cols, rows


def visible_gt_row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("NCT", "") or ""),
        str(row.get("PubMed ID", "") or ""),
        str(row.get("Trial name", "") or ""),
        str(row.get("source_value", "") or ""),
    )


def candidate_row_key(row: Dict[str, Any], source_column: str) -> Tuple[str, str, str, str]:
    return (
        str(row.get("NCT", "") or ""),
        str(row.get("PubMed ID", "") or ""),
        str(row.get("Trial name", "") or ""),
        str(row.get(source_column, "") or ""),
    )


def build_column_text(source_column: str, value: Any) -> str:
    return f"{source_column}: {canonical_value(value)}"


def build_full_row_text(row: Dict[str, Any]) -> str:
    items = []
    for key, value in row.items():
        if key == "__rowid__":
            continue
        items.append((key, canonical_value(value)))
    return build_row_text(dict(items))


def reciprocal_rank(ranks: Sequence[int]) -> float:
    if not ranks:
        return 0.0
    return 1.0 / float(min(ranks))


def load_sample_questions(manifest_csv: Path, limit: int) -> List[Dict[str, Any]]:
    rows = read_csv_rows(manifest_csv)
    out: List[Dict[str, Any]] = []
    for manifest_row in rows:
        if (manifest_row.get("status") or "").strip() != "ok":
            continue
        question_dir = Path(manifest_row["question_dir"])
        metadata = read_json(question_dir / "metadata.json")
        gt_rows = read_csv_rows(question_dir / "ground_truth_table.csv")
        out.append(
            {
                "item_id": manifest_row["item_id"],
                "csv_row_number": int(manifest_row["csv_row_number"]),
                "question_dir": str(question_dir),
                "question": metadata["question"],
                "column_used": metadata["column_used"],
                "expected_keys": list(metadata["expected_keys"]),
                "ground_truth_rows": gt_rows,
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip() or "question_table_copy_cosine_100")
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"qtable_cosine_{run_dir.name}")

    logger.info("Manifest CSV: %s", manifest_csv)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)

    sample_questions = load_sample_questions(manifest_csv, args.limit)
    sample_manifest_rows = [
        {
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": item["question"],
            "column_used": item["column_used"],
            "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
            "question_dir": item["question_dir"],
            "ground_truth_row_count": len(item["ground_truth_rows"]),
        }
        for item in sample_questions
    ]
    write_csv(run_dir / "sample_manifest.csv", sample_manifest_rows)

    conn = sqlite3.connect(db_path)
    try:
        table_cols, full_table_rows = fetch_full_table(conn, args.table_name)
    finally:
        conn.close()

    model_specs = model_specs_from_args(args)
    embedders: Dict[str, LocalEmbedder] = {}
    for name, model_path, instruction_prefix in model_specs:
        logger.info("Loading embedder %s from %s", name, model_path)
        embedders[name] = LocalEmbedder(model_path, device=args.device, instruction_prefix=instruction_prefix)

    row_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []

    for item in sample_questions:
        question = item["question"]
        source_column = item["column_used"]
        if source_column not in table_cols:
            logger.info("Skipping %s because source column %s is not in schema", item["item_id"], source_column)
            continue

        question_slug = sanitize_name(f"{item['item_id']}__{question}")[:180]
        question_dir = run_dir / "questions" / question_slug
        question_dir.mkdir(parents=True, exist_ok=True)

        table_copy_path = question_dir / "table_copy.csv"
        write_csv(table_copy_path, full_table_rows)
        write_csv(question_dir / "ground_truth_table.csv", item["ground_truth_rows"])
        write_json(
            question_dir / "metadata.json",
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "expected_keys": item["expected_keys"],
                "source_table_copy_csv": str(table_copy_path),
                "original_question_dir": item["question_dir"],
            },
        )

        relevant_keys = {visible_gt_row_key(row) for row in item["ground_truth_rows"]}
        if not relevant_keys:
            continue

        candidate_rows = []
        for row in full_table_rows:
            source_value = row.get(source_column)
            if source_value is None or str(source_value).strip() == "":
                continue
            candidate_rows.append(row)

        if not candidate_rows:
            continue

        query_text = question
        column_candidate_texts = [build_column_text(source_column, row.get(source_column)) for row in candidate_rows]
        row_candidate_texts = [build_full_row_text(row) for row in candidate_rows]
        tuple_candidate_texts = [build_tuple_text(question, row) for row in candidate_rows]

        question_metric_rows: List[Dict[str, Any]] = []
        topk_rows_out: List[Dict[str, Any]] = []
        for model_name, embedder in embedders.items():
            query_vec = embedder.encode([query_text], batch_size=1)[0]
            for view_name, texts in (
                ("column", column_candidate_texts),
                ("row", row_candidate_texts),
                ("tuple", tuple_candidate_texts),
            ):
                matrix = embedder.encode(texts, batch_size=args.batch_size)
                scores = np.matmul(matrix, query_vec)
                ranked_indices = np.argsort(-scores)
                relevant_ranks: List[int] = []
                for rank_pos, idx in enumerate(ranked_indices, start=1):
                    key = candidate_row_key(candidate_rows[int(idx)], source_column)
                    if key in relevant_keys:
                        relevant_ranks.append(rank_pos)

                hit_at_1 = 1.0 if any(rank <= 1 for rank in relevant_ranks) else 0.0
                hit_at_5 = 1.0 if any(rank <= 5 for rank in relevant_ranks) else 0.0
                hit_at_10 = 1.0 if any(rank <= 10 for rank in relevant_ranks) else 0.0
                mrr = reciprocal_rank(relevant_ranks)
                best_relevant_rank = min(relevant_ranks) if relevant_ranks else 0

                top_idx = int(ranked_indices[0])
                top_row = candidate_rows[top_idx]
                top_payload = top_row.get(source_column)
                actual_payloads = [row.get("derived_expected_llm_response", "") for row in item["ground_truth_rows"]]
                naive_exact = max(exact_match(top_payload, payload) for payload in actual_payloads) if actual_payloads else 0.0

                question_metric_rows.append(
                    {
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": question,
                        "column_used": source_column,
                        "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                        "model_name": model_name,
                        "view_name": view_name,
                        "relevant_row_count": len(relevant_keys),
                        "candidate_row_count": len(candidate_rows),
                        "hit_at_1": hit_at_1,
                        "hit_at_5": hit_at_5,
                        "hit_at_10": hit_at_10,
                        "mrr": mrr,
                        "best_relevant_rank": best_relevant_rank,
                        "top1_cosine_score": float(scores[top_idx]),
                        "top1_rowid": top_row.get("__rowid__"),
                        "top1_source_value": canonical_value(top_row.get(source_column)),
                        "naive_source_as_answer_exact": naive_exact,
                    }
                )

                for rank_pos, idx in enumerate(ranked_indices[: max(1, args.top_k)], start=1):
                    row = candidate_rows[int(idx)]
                    key = candidate_row_key(row, source_column)
                    topk_rows_out.append(
                        {
                            "item_id": item["item_id"],
                            "csv_row_number": item["csv_row_number"],
                            "question": question,
                            "column_used": source_column,
                            "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                            "model_name": model_name,
                            "view_name": view_name,
                            "rank": rank_pos,
                            "cosine_score": float(scores[int(idx)]),
                            "is_relevant": 1 if key in relevant_keys else 0,
                            "rowid": row.get("__rowid__"),
                            "NCT": row.get("NCT"),
                            "PubMed ID": row.get("PubMed ID"),
                            "Trial name": row.get("Trial name"),
                            "source_value": row.get(source_column),
                            "candidate_row_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
                        }
                    )

        write_csv(question_dir / "topk_retrievals.csv", topk_rows_out)
        question_rows.extend(question_metric_rows)
        row_rows.extend(topk_rows_out)

    question_results_csv = run_dir / "all_question_results.csv"
    row_predictions_csv = run_dir / "row_level_predictions.csv"
    write_csv(question_results_csv, question_rows)
    write_csv(row_predictions_csv, row_rows)

    summary_rows: List[Dict[str, Any]] = []
    by_model_view: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        by_model_view[(str(row["model_name"]), str(row["view_name"]))].append(row)
    for (model_name, view_name), rows_out in sorted(by_model_view.items()):
        summary_rows.append(
            {
                "model_name": model_name,
                "view_name": view_name,
                "question_count": len(rows_out),
                "avg_hit_at_1": avg_numeric(rows_out, "hit_at_1"),
                "avg_hit_at_5": avg_numeric(rows_out, "hit_at_5"),
                "avg_hit_at_10": avg_numeric(rows_out, "hit_at_10"),
                "avg_mrr": avg_numeric(rows_out, "mrr"),
                "avg_top1_cosine_score": avg_numeric(rows_out, "top1_cosine_score"),
                "avg_naive_source_as_answer_exact": avg_numeric(rows_out, "naive_source_as_answer_exact"),
            }
        )
    baseline_summary_csv = run_dir / "baseline_summary.csv"
    write_csv(baseline_summary_csv, summary_rows)

    summary = {
        "manifest_csv": str(manifest_csv),
        "db_path": str(db_path),
        "sample_question_count": len(sample_questions),
        "question_result_count": len(question_rows),
        "topk_row_count": len(row_rows),
        "outputs": {
            "sample_manifest_csv": str(run_dir / "sample_manifest.csv"),
            "question_results_csv": str(question_results_csv),
            "row_predictions_csv": str(row_predictions_csv),
            "baseline_summary_csv": str(baseline_summary_csv),
        },
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote sample manifest: %s", run_dir / "sample_manifest.csv")
    logger.info("Wrote question results: %s", question_results_csv)
    logger.info("Wrote row predictions: %s", row_predictions_csv)
    logger.info("Wrote summary: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
