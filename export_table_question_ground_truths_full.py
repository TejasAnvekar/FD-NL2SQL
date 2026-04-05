#!/usr/bin/env python3
"""Export question-level ground truths for rows whose expected keys are all Table."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence

from execute_table_derivation_sqls import parse_jsonish, register_udfs, slugify, write_csv, write_json
from generate_table_derivation_sqls import build_base_sql, derivation_expression


DEFAULT_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv"
DEFAULT_SUMMARY = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_expected_keys_summary_rettable.csv"
DEFAULT_DB = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_OUTDIR = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full question-level ground truths for Table-only expected key rows.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--summary_csv", default=DEFAULT_SUMMARY)
    parser.add_argument("--db_path", default=DEFAULT_DB)
    parser.add_argument("--output_dir", default=DEFAULT_OUTDIR)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_question_sql(sql_query: str, column_used: str, keys: Sequence[str]) -> str:
    base_sql = build_base_sql(sql_query, column_used).rstrip(";")
    exprs: List[str] = []
    for key in keys:
        expr, _ = derivation_expression(key, column_used)
        exprs.append(f"{expr} AS \"{key}\"")
    return (
        f"WITH base AS (\n{base_sql}\n)\n"
        f"SELECT \"NCT\", \"PubMed ID\", \"Trial name\", source_value, {', '.join(exprs)}\n"
        f"FROM base;"
    )


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    questions_dir = output_dir / "questions"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    rows = load_rows(csv_path)
    summary_rows = load_rows(summary_csv)
    key_labels = {
        (row.get("expected_key") or "").strip(): (row.get("Table or Retrieval") or "").strip()
        for row in summary_rows
    }

    conn = sqlite3.connect(db_path)
    register_udfs(conn)
    try:
        manifest_rows: List[Dict[str, Any]] = []
        exported_count = 0
        skipped_count = 0
        included_question_count = 0

        for csv_row_number, row in enumerate(rows, start=2):
            question = (row.get("natural_language_query") or "").strip()
            sql_query = (row.get("sql_query") or "").strip()
            column_used = (row.get("column_used") or "").strip()
            payload_text = (row.get("expected_llm_response") or "").strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except Exception:
                continue
            if not isinstance(payload, dict) or not payload:
                continue
            keys = [str(k).strip() for k in payload.keys() if str(k).strip()]
            if not keys:
                continue
            if any(key_labels.get(key) != "Table" for key in keys):
                continue

            included_question_count += 1
            item_id = f"row_{csv_row_number}"
            question_dir = questions_dir / f"{slugify('_'.join(keys), 40)}__{item_id}__{slugify(question)}"
            status = "ok"
            error = ""

            try:
                ground_truth_sql = build_question_sql(sql_query, column_used, keys)
                cur = conn.execute(ground_truth_sql.rstrip(";"))
                result_cols = [d[0] for d in cur.description] if cur.description else []
                result_rows = cur.fetchall() if cur.description else []

                export_rows: List[Dict[str, Any]] = []
                for row_index, values in enumerate(result_rows, start=1):
                    obj = {result_cols[idx]: values[idx] for idx in range(len(result_cols))}
                    response_obj = {key: parse_jsonish(obj.get(key)) for key in keys}
                    row_out: Dict[str, Any] = {"row_index": row_index}
                    for col in result_cols:
                        row_out[col] = obj.get(col)
                    row_out["derived_expected_llm_response"] = json.dumps(response_obj, ensure_ascii=False)
                    export_rows.append(row_out)

                fieldnames = ["row_index"] + result_cols + ["derived_expected_llm_response"]
                ground_truth_csv = question_dir / "ground_truth_table.csv"
                write_csv(ground_truth_csv, fieldnames, export_rows)
                write_json(
                    question_dir / "metadata.json",
                    {
                        "item_id": item_id,
                        "csv_row_number": csv_row_number,
                        "question": question,
                        "column_used": column_used,
                        "expected_keys": keys,
                        "expected_llm_response": payload,
                        "sql_query": sql_query,
                        "ground_truth_sql": ground_truth_sql,
                        "ground_truth_table_csv": str(ground_truth_csv),
                        "row_count": len(export_rows),
                    },
                )
                exported_count += 1
            except Exception as exc:
                status = "skipped"
                error = str(exc)
                skipped_count += 1
                write_json(
                    question_dir / "metadata.json",
                    {
                        "item_id": item_id,
                        "csv_row_number": csv_row_number,
                        "question": question,
                        "column_used": column_used,
                        "expected_keys": keys,
                        "expected_llm_response": payload,
                        "sql_query": sql_query,
                        "status": status,
                        "error": error,
                    },
                )

            manifest_rows.append(
                {
                    "item_id": item_id,
                    "csv_row_number": csv_row_number,
                    "question": question,
                    "expected_keys_json": json.dumps(keys, ensure_ascii=False),
                    "status": status,
                    "error": error,
                    "question_dir": str(question_dir),
                    "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv") if status == "ok" else "",
                }
            )
    finally:
        conn.close()

    write_csv(
        manifest_path,
        ["item_id", "csv_row_number", "question", "expected_keys_json", "status", "error", "question_dir", "ground_truth_table_csv"],
        manifest_rows,
    )
    write_json(
        summary_path,
        {
            "input_csv": str(csv_path),
            "summary_csv": str(summary_csv),
            "db_path": str(db_path),
            "output_dir": str(output_dir),
            "included_question_count": included_question_count,
            "exported_count": exported_count,
            "skipped_count": skipped_count,
            "manifest_csv": str(manifest_path),
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "included_question_count": included_question_count,
                "exported_count": exported_count,
                "skipped_count": skipped_count,
                "manifest_csv": str(manifest_path),
                "summary_json": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
