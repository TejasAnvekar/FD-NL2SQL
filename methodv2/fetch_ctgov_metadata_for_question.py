#!/usr/bin/env python3
"""Fetch ClinicalTrials.gov metadata for studies referenced by a CSV question row."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinicaltrials_api import fetch_study_by_nct, summarize_study  # noqa: E402
from run_hidden_column_sql_eval import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, load_csv_rows, make_run_dir  # noqa: E402
from utils import quote_ident, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "For a given CSV row number, execute its SQL with NCT/Trial name added, then fetch "
            "ClinicalTrials.gov metadata for each returned NCT."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--csv_row_number", type=int, required=True)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--gt_sql_key", default="sql_query")
    ap.add_argument("--column_used_key", default="column_used")
    ap.add_argument("--run_root", default=str(PROJECT_ROOT / "methodv2" / "runs" / "ctgov_fetches"))
    ap.add_argument("--run_name", default="")
    ap.add_argument("--timeout", type=float, default=60.0)
    return ap.parse_args()


def build_select_variant(sql: str, select_exprs: Sequence[str]) -> str:
    import re

    sql0 = (sql or "").strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    return f"SELECT {', '.join(select_exprs)} {sql0[match.start():]};"


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.execute((sql or "").strip().rstrip(";"))
    cols = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall() if cur.description else []
    return cols, rows


def rows_to_objects(cols: Sequence[str], rows: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        obj: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            obj[col] = row[idx] if idx < len(row) else None
        out.append(obj)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip() or f"row_{args.csv_row_number}")

    rows = load_csv_rows(csv_path)
    if args.csv_row_number < 2:
        raise ValueError("csv_row_number must be 2 or greater because row 1 is the header")
    row_index = args.csv_row_number - 2
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"csv_row_number {args.csv_row_number} is out of bounds")
    row = rows[row_index]

    question = (row.get(args.question_key) or "").strip()
    gt_sql = (row.get(args.gt_sql_key) or "").strip()
    source_column = (row.get(args.column_used_key) or "").strip()
    if not gt_sql:
        raise ValueError(f"Row {args.csv_row_number} has no sql_query")

    select_exprs = [quote_ident("NCT"), quote_ident("Trial name")]
    if source_column and source_column not in {"NCT", "Trial name"}:
        select_exprs.append(quote_ident(source_column))
    enriched_sql = build_select_variant(gt_sql, select_exprs)

    conn = sqlite3.connect(db_path)
    try:
        result_cols, result_rows = execute_query(conn, enriched_sql)
    finally:
        conn.close()

    result_objects = rows_to_objects(result_cols, result_rows)
    write_csv(run_dir / "row_results.csv", result_objects)

    summary_rows: List[Dict[str, Any]] = []
    studies_dir = run_dir / "studies"
    seen_ncts = set()
    for row_obj in result_objects:
        nct = str(row_obj.get("NCT") or "").strip()
        if not nct or nct in seen_ncts:
            continue
        seen_ncts.add(nct)
        trial_name = str(row_obj.get("Trial name") or "").strip()
        study_dir = studies_dir / nct
        study_dir.mkdir(parents=True, exist_ok=True)

        raw_obj = fetch_study_by_nct(nct, timeout=args.timeout)
        summary_obj = summarize_study(raw_obj)
        write_json(study_dir / "raw.json", raw_obj)
        write_json(study_dir / "summary.json", summary_obj)

        summary_rows.append(
            {
                "NCT": nct,
                "trial_name_from_db": trial_name,
                "source_column": source_column,
                "source_value": row_obj.get(source_column) if source_column else None,
                "ctgov_brief_title": summary_obj.get("brief_title"),
                "ctgov_acronym": summary_obj.get("acronym"),
                "overall_status": summary_obj.get("overall_status"),
                "phases_json": json.dumps(summary_obj.get("phases") or [], ensure_ascii=False),
                "conditions_json": json.dumps(summary_obj.get("conditions") or [], ensure_ascii=False),
                "candidate_control_arms_json": json.dumps(summary_obj.get("candidate_control_arms") or [], ensure_ascii=False),
                "interventions_json": json.dumps(summary_obj.get("interventions") or [], ensure_ascii=False),
                "reference_pmids_json": json.dumps(summary_obj.get("reference_pmids") or [], ensure_ascii=False),
                "study_dir": str(study_dir),
            }
        )

    write_csv(run_dir / "ctgov_summary.csv", summary_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "csv_path": str(csv_path),
            "db_path": str(db_path),
            "csv_row_number": args.csv_row_number,
            "question": question,
            "source_column": source_column,
            "gt_sql": gt_sql,
            "enriched_sql": enriched_sql,
            "row_results_csv": str(run_dir / "row_results.csv"),
            "ctgov_summary_csv": str(run_dir / "ctgov_summary.csv"),
            "studies_dir": str(studies_dir),
        },
    )


if __name__ == "__main__":
    main()
