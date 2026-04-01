#!/usr/bin/env python3
"""Hidden-column SQL generation experiment runner for methodv2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_schema_value_hints, quote_ident, setup_logger, write_json, write_jsonl

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


DEFAULT_MODEL_PATH = (
    "/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/"
    "snapshots/005ad3404e59d6023443cb575daa05336842228a"
)
DEFAULT_CSV_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv"
DEFAULT_DB_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_RUN_ROOT = str(PROJECT_ROOT / "methodv2" / "runs")
EVAL_SCRIPT_PATH = PROJECT_ROOT / "eval_run_baselines_v2.py"


@dataclass
class QuestionItem:
    item_id: str
    csv_row_number: int
    question: str
    gt_sql: str
    removed_column: str
    expected_llm_response: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run hidden-column SQL generation experiments grouped by final_column "
            "and evaluate them with eval_run_baselines_v2.py."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--gt_sql_key", default="sql_query")
    ap.add_argument("--final_column_key", default="final_column")
    ap.add_argument("--expected_response_key", default="expected_llm_response")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="", help="Optional fixed run directory name.")
    ap.add_argument("--limit", type=int, default=0, help="0 means all eligible rows.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--trust_remote_code", type=int, default=1)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_eval_rows", type=int, default=10000)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare reduced DBs, prompts, and GT files but skip model generation and evaluation.",
    )
    return ap.parse_args()


def sanitize_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def make_run_dir(run_root: Path, run_name: str) -> Path:
    if run_name:
        run_dir = run_root / run_name
    else:
        run_dir = run_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def fetch_schema(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [row[1] for row in rows]


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_final_column(raw_value: str, schema_cols: Sequence[str]) -> Optional[str]:
    value = (raw_value or "").strip()
    if not value or value == "no_match":
        return None
    return value if value in set(schema_cols) else None


def collect_items(
    rows: Sequence[Dict[str, str]],
    *,
    schema_cols: Sequence[str],
    question_key: str,
    gt_sql_key: str,
    final_column_key: str,
    expected_response_key: str,
    limit: int,
) -> Tuple[List[QuestionItem], Dict[str, int]]:
    items: List[QuestionItem] = []
    skipped = Counter()
    schema_set = set(schema_cols)

    for idx, row in enumerate(rows, start=2):
        removed_column = pick_final_column(row.get(final_column_key, ""), schema_set)
        if removed_column is None:
            skipped["final_column_not_usable"] += 1
            continue

        question = (row.get(question_key) or "").strip()
        gt_sql = (row.get(gt_sql_key) or "").strip()
        if not question:
            skipped["missing_question"] += 1
            continue
        if not gt_sql:
            skipped["missing_gt_sql"] += 1
            continue

        item = QuestionItem(
            item_id=f"row_{idx}",
            csv_row_number=idx,
            question=question,
            gt_sql=gt_sql,
            removed_column=removed_column,
            expected_llm_response=(row.get(expected_response_key) or "").strip(),
        )
        items.append(item)
        if limit and len(items) >= limit:
            break

    return items, dict(skipped)


def create_reduced_db(
    *,
    source_db: Path,
    output_db: Path,
    table_name: str,
    removed_column: str,
) -> List[str]:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, output_db)

    conn = sqlite3.connect(output_db)
    try:
        cols = fetch_schema(conn, table_name)
        if removed_column not in cols:
            raise ValueError(f"Column {removed_column!r} does not exist in table {table_name!r}")

        remaining_cols = [col for col in cols if col != removed_column]
        temp_table = f"{table_name}__reduced"
        select_cols = ", ".join(quote_ident(col) for col in remaining_cols)
        conn.execute(f'DROP TABLE IF EXISTS "{temp_table}"')
        conn.execute(f'CREATE TABLE "{temp_table}" AS SELECT {select_cols} FROM "{table_name}"')
        conn.execute(f'DROP TABLE "{table_name}"')
        conn.execute(f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}"')
        conn.commit()
        return remaining_cols
    finally:
        conn.close()


def render_schema_hints(schema_cols: Sequence[str], hints: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for col in schema_cols:
        hint = hints.get(col)
        if not hint:
            continue
        values = hint.get("values") or []
        if not values:
            continue
        values_inline = ", ".join(str(v) for v in values)
        shown = int(hint.get("shown", len(values)))
        total = int(hint.get("total_distinct", shown))
        if bool(hint.get("truncated", False)):
            lines.append(f'- "{col}": {values_inline} (showing {shown}/{total})')
        else:
            lines.append(f'- "{col}": {values_inline}')
    if not lines:
        return ""
    return "Observed values in the visible schema:\n" + "\n".join(lines)


def build_hidden_column_prompt(
    *,
    question: str,
    table_name: str,
    visible_columns: Sequence[str],
    schema_hints_text: str,
) -> str:
    schema_inline = ", ".join(f'"{col}"' for col in visible_columns)
    prompt = "\n".join(
        [
            f'You are a SQL generator. Write one SQLite SELECT query over "{table_name}".',
            "One semantically useful column has been intentionally hidden from the visible schema.",
            "Infer the answer using only the visible columns and their observed values.",
            "Rules:",
            "- Output ONLY the SQL query.",
            "- The SQL MUST start with SELECT or WITH.",
            f"- Use only columns from this visible schema: {schema_inline}",
            "- Put double quotes around EVERY column name exactly as shown in the visible schema.",
            "- Use single quotes for string literals.",
            "- Do not reference any hidden, missing, or invented column.",
            "- If inference is needed, ground it in visible columns or visible values only.",
        ]
    )
    if schema_hints_text:
        prompt += "\n\n" + schema_hints_text
    prompt += f"\n\nQuestion: {question}\nSQL:"
    return prompt


def extract_sql_candidate(text: str) -> str:
    if not text:
        return ""
    candidate = text.strip()
    candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", candidate).strip()
    candidate = re.sub(r"\s*```$", "", candidate).strip()

    match = re.search(r"\b(SELECT|WITH)\b", candidate, flags=re.IGNORECASE)
    if match:
        candidate = candidate[match.start() :].strip()

    if ";" in candidate:
        candidate = candidate.split(";", 1)[0].strip() + ";"
    return candidate


def chunked(seq: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int]]:
    size = max(1, int(batch_size))
    start = 0
    while start < len(seq):
        end = min(len(seq), start + size)
        yield start, end
        start = end


def init_vllm(args: argparse.Namespace) -> Tuple["LLM", "SamplingParams"]:
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vllm is not installed/importable in this environment.")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        trust_remote_code=bool(args.trust_remote_code),
        dtype=args.dtype,
        seed=int(args.seed),
    )
    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    return llm, sampling


def generate_sql_predictions(
    *,
    llm: "LLM",
    sampling: "SamplingParams",
    prompts: Sequence[str],
) -> List[str]:
    outputs = llm.generate(list(prompts), sampling)
    texts: List[str] = []
    for out in outputs:
        text = (out.outputs[0].text or "").strip() if out.outputs else ""
        texts.append(text)
    return texts


def run_evaluator(
    *,
    pred_path: Path,
    gt_path: Path,
    db_path: Path,
    output_json: Path,
    question_key: str,
    gt_sql_key: str,
    max_rows: int,
) -> None:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT_PATH),
        "--pred_path",
        str(pred_path),
        "--gt_path",
        str(gt_path),
        "--db_path",
        str(db_path),
        "--output_json",
        str(output_json),
        "--pred_format",
        "jsonl",
        "--gt_format",
        "jsonl",
        "--id_key",
        "item_id",
        "--pred_sql_key",
        "pred_sql",
        "--gt_sql_key",
        gt_sql_key,
        "--question_key",
        question_key,
        "--max_rows",
        str(max_rows),
    ]
    subprocess.run(cmd, check=True)


def weighted_average(group_summaries: Sequence[Dict[str, Any]], key: str, weight_key: str = "evaluated_items") -> Optional[float]:
    num = 0.0
    den = 0.0
    for summary in group_summaries:
        value = summary.get(key)
        weight = summary.get(weight_key)
        if isinstance(value, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
            num += float(value) * float(weight)
            den += float(weight)
    return (num / den) if den else None


def aggregate_eval_summaries(group_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    group_summaries = [gr["summary"] for gr in group_results if isinstance(gr.get("summary"), dict)]

    summed_keys = [
        "total_gt_items",
        "total_pred_items",
        "evaluated_items",
        "pred_with_sql",
        "pred_exec_ok",
        "gt_exec_ok",
        "exec_eval_ok",
        "exec_exact_match",
        "pred_limit_no_order",
        "gt_limit_no_order",
    ]
    aggregate: Dict[str, Any] = {}
    for key in summed_keys:
        aggregate[key] = int(sum(int(s.get(key, 0) or 0) for s in group_summaries))

    ratio_keys = [
        "avg_sql_ast_similarity",
        "avg_precision",
        "avg_recall",
        "avg_f1",
        "avg_row_jaccard",
        "avg_normalization_factor",
        "avg_column_alignment_score",
        "avg_chrf",
        "avg_rouge_l_f1",
        "avg_bertscore_f1",
    ]
    for key in ratio_keys:
        aggregate[key] = weighted_average(group_summaries, key)

    exec_ready = aggregate.get("exec_eval_ok", 0)
    exact_match = aggregate.get("exec_exact_match", 0)
    aggregate["exec_exact_match_rate"] = (float(exact_match) / float(exec_ready)) if exec_ready else 0.0
    return aggregate


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip())
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"methodv2_{run_dir.name}")

    logger.info("CSV path: %s", csv_path)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Model path: %s", args.model_path)
    logger.info("Dry run: %s", bool(args.dry_run))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    raw_rows = load_csv_rows(csv_path)
    conn = sqlite3.connect(db_path)
    try:
        original_schema = fetch_schema(conn, args.table_name)
    finally:
        conn.close()

    items, skipped = collect_items(
        raw_rows,
        schema_cols=original_schema,
        question_key=args.question_key,
        gt_sql_key=args.gt_sql_key,
        final_column_key=args.final_column_key,
        expected_response_key=args.expected_response_key,
        limit=args.limit,
    )
    grouped_items: Dict[str, List[QuestionItem]] = defaultdict(list)
    for item in items:
        grouped_items[item.removed_column].append(item)

    logger.info("Eligible rows: %d", len(items))
    logger.info("Skipped rows: %s", skipped)
    logger.info("Groups: %d", len(grouped_items))
    for column_name, group in sorted(grouped_items.items()):
        logger.info("  %s -> %d rows", column_name, len(group))

    meta = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "table_name": args.table_name,
        "model_path": args.model_path,
        "dry_run": bool(args.dry_run),
        "eligible_items": len(items),
        "skipped": skipped,
        "groups": {col: len(group) for col, group in sorted(grouped_items.items())},
    }
    write_json(run_dir / "run_meta.json", meta)

    all_gt_rows = [
        {
            "item_id": item.item_id,
            args.question_key: item.question,
            args.gt_sql_key: item.gt_sql,
            "removed_column": item.removed_column,
            "csv_row_number": item.csv_row_number,
            "expected_llm_response": item.expected_llm_response,
        }
        for item in items
    ]
    write_jsonl(run_dir / "all_gt.jsonl", all_gt_rows)

    if not items:
        logger.warning("No eligible items found. Nothing to run.")
        return

    llm = None
    sampling = None
    if not args.dry_run:
        llm, sampling = init_vllm(args)

    group_results: List[Dict[str, Any]] = []
    all_pred_rows: List[Dict[str, Any]] = []

    for removed_column, group in sorted(grouped_items.items()):
        safe_group = sanitize_name(removed_column)
        group_dir = run_dir / f"group__{safe_group}"
        group_dir.mkdir(parents=True, exist_ok=True)

        reduced_db_path = group_dir / f"{safe_group}.db"
        visible_schema = create_reduced_db(
            source_db=db_path,
            output_db=reduced_db_path,
            table_name=args.table_name,
            removed_column=removed_column,
        )
        logger.info("Prepared reduced DB for group=%s at %s", removed_column, reduced_db_path)

        reduced_conn = sqlite3.connect(reduced_db_path)
        try:
            schema_hints = fetch_schema_value_hints(reduced_conn, args.table_name, visible_schema)
        finally:
            reduced_conn.close()
        schema_hints_text = render_schema_hints(visible_schema, schema_hints)

        gt_rows = []
        prompt_rows = []
        prompts: List[str] = []
        for item in group:
            prompt = build_hidden_column_prompt(
                question=item.question,
                table_name=args.table_name,
                visible_columns=visible_schema,
                schema_hints_text=schema_hints_text,
            )
            gt_rows.append(
                {
                    "item_id": item.item_id,
                    args.question_key: item.question,
                    args.gt_sql_key: item.gt_sql,
                    "removed_column": removed_column,
                    "csv_row_number": item.csv_row_number,
                    "expected_llm_response": item.expected_llm_response,
                }
            )
            prompt_rows.append(
                {
                    "item_id": item.item_id,
                    "removed_column": removed_column,
                    "question": item.question,
                    "prompt": prompt,
                }
            )
            prompts.append(prompt)

        gt_path = group_dir / "gt.jsonl"
        prompts_path = group_dir / "prompts.jsonl"
        pred_path = group_dir / "pred.jsonl"
        eval_path = group_dir / "eval.json"

        write_jsonl(gt_path, gt_rows)
        write_jsonl(prompts_path, prompt_rows)

        group_result: Dict[str, Any] = {
            "removed_column": removed_column,
            "group_dir": str(group_dir),
            "reduced_db_path": str(reduced_db_path),
            "items": len(group),
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "eval_path": str(eval_path),
            "dry_run": bool(args.dry_run),
        }

        if args.dry_run:
            logger.info("Dry run enabled; skipping model generation/eval for group=%s", removed_column)
            group_results.append(group_result)
            continue

        assert llm is not None and sampling is not None
        pred_rows: List[Dict[str, Any]] = []
        for start, end in chunked(prompts, args.batch_size):
            batch_prompts = prompts[start:end]
            batch_items = group[start:end]
            batch_texts = generate_sql_predictions(llm=llm, sampling=sampling, prompts=batch_prompts)
            for item, raw_text in zip(batch_items, batch_texts):
                pred_row = {
                    "item_id": item.item_id,
                    "question_used": item.question,
                    "removed_column": removed_column,
                    "csv_row_number": item.csv_row_number,
                    "pred_sql": extract_sql_candidate(raw_text),
                    "raw_text": raw_text,
                    "model_path": args.model_path,
                }
                pred_rows.append(pred_row)
                all_pred_rows.append(pred_row)

        write_jsonl(pred_path, pred_rows)
        logger.info("Generated %d predictions for group=%s", len(pred_rows), removed_column)

        run_evaluator(
            pred_path=pred_path,
            gt_path=gt_path,
            db_path=reduced_db_path,
            output_json=eval_path,
            question_key=args.question_key,
            gt_sql_key=args.gt_sql_key,
            max_rows=args.max_eval_rows,
        )
        eval_obj = json.load(open(eval_path, "r", encoding="utf-8"))
        group_result["summary"] = eval_obj.get("summary", {})
        group_results.append(group_result)
        logger.info("Evaluated group=%s summary=%s", removed_column, group_result["summary"])

    if all_pred_rows:
        write_jsonl(run_dir / "all_pred.jsonl", all_pred_rows)

    summary = {
        "meta": meta,
        "group_results": group_results,
        "aggregate_summary": aggregate_eval_summaries(group_results) if not args.dry_run else None,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote summary to %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
