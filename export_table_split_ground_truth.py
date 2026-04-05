#!/usr/bin/env python3
"""Export per-question ground-truth tables for the table split CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches_table_rows.csv"
DEFAULT_DB = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_OUTDIR = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_rows_ground_truth"
PREFERRED_CONTEXT_COLUMNS = ["NCT", "PubMed ID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a ground_truth_table.csv for every row in the table split CSV."
    )
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--db_path", default=DEFAULT_DB)
    parser.add_argument("--table_name", default="clinical_trials")
    parser.add_argument("--question_key", default="natural_language_query")
    parser.add_argument("--sql_key", default="sql_query")
    parser.add_argument("--ground_truth_column_key", default="ground_truth_column")
    parser.add_argument("--output_dir", default=DEFAULT_OUTDIR)
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def slugify(text: str, max_chars: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "question"
    return slug[:max_chars].rstrip("_") or "question"


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


def stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.execute((sql or "").strip().rstrip(";"))
    cols = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall() if cur.description else []
    return cols, rows


def build_select_variant(sql: str, select_exprs: Sequence[str]) -> str:
    sql0 = (sql or "").strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    return f"SELECT {', '.join(select_exprs)} {sql0[match.start():]};"


def unique_in_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_ground_truth_targets(text: str) -> List[Tuple[str, str]]:
    raw = (text or "").strip()
    if not raw or raw == "no_match":
        return []
    if " | " in raw and " -> " in raw:
        pairs: List[Tuple[str, str]] = []
        for part in raw.split(" | "):
            if " -> " not in part:
                continue
            output_name, db_column = part.split(" -> ", 1)
            output_name = output_name.strip()
            db_column = db_column.strip()
            if output_name and db_column and db_column != "no_match":
                pairs.append((output_name, db_column))
        return pairs
    return [(raw, raw)]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_cell(row.get(key)) for key in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    questions_dir = output_dir / "questions"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    conn = sqlite3.connect(db_path)
    try:
        schema_cols = [row[1] for row in conn.execute(f"PRAGMA table_info({quote_ident(args.table_name)})")]

        manifest_rows: List[Dict[str, Any]] = []
        exported_count = 0
        skipped_count = 0

        for csv_row_number, row in enumerate(rows, start=2):
            question = (row.get(args.question_key) or "").strip()
            sql_query = (row.get(args.sql_key) or "").strip()
            ground_truth_text = (row.get(args.ground_truth_column_key) or "").strip()
            target_pairs = parse_ground_truth_targets(ground_truth_text)
            item_id = f"row_{csv_row_number}"
            question_dir = questions_dir / f"{slugify(ground_truth_text or 'no_match', 60)}__{item_id}__{slugify(question)}"

            status = "ok"
            error = ""
            visible_cols: List[str] = []
            visible_rows: List[Tuple[Any, ...]] = []
            target_columns: List[str] = []
            augmented_sql = ""

            try:
                if not sql_query:
                    raise ValueError("missing sql_query")
                if not target_pairs:
                    raise ValueError("no usable ground_truth_column target")

                invalid_targets = [db_col for _, db_col in target_pairs if db_col not in schema_cols]
                if invalid_targets:
                    raise ValueError(f"ground truth targets not in schema: {invalid_targets}")

                visible_cols, visible_rows = execute_query(conn, sql_query)
                target_columns = []
                for _, db_col in target_pairs:
                    if db_col not in target_columns:
                        target_columns.append(db_col)

                context_cols = [col for col in PREFERRED_CONTEXT_COLUMNS if col in schema_cols]
                combined_cols = unique_in_order(context_cols + list(visible_cols))
                for db_col in target_columns:
                    if db_col not in combined_cols:
                        combined_cols.append(db_col)

                augmented_sql = build_select_variant(sql_query, [quote_ident(col) for col in combined_cols])
                aug_cols, aug_rows = execute_query(conn, augmented_sql)

                output_fieldnames = ["row_index"] + unique_in_order(context_cols + list(visible_cols))
                for output_name, _ in target_pairs:
                    if output_name not in output_fieldnames:
                        output_fieldnames.append(output_name)

                export_rows: List[Dict[str, Any]] = []
                col_index = {col: idx for idx, col in enumerate(aug_cols)}
                for row_index, values in enumerate(aug_rows, start=1):
                    row_out: Dict[str, Any] = {"row_index": row_index}
                    for col in visible_cols:
                        if col in col_index:
                            row_out[col] = canonical_value(values[col_index[col]])
                    for output_name, db_col in target_pairs:
                        row_out[output_name] = canonical_value(values[col_index[db_col]]) if db_col in col_index else None
                    export_rows.append(row_out)

                ground_truth_csv = question_dir / "ground_truth_table.csv"
                write_csv(ground_truth_csv, output_fieldnames, export_rows)
                write_json(
                    question_dir / "metadata.json",
                    {
                        "item_id": item_id,
                        "csv_row_number": csv_row_number,
                        "question": question,
                        "sql_query": sql_query,
                        "ground_truth_column": ground_truth_text,
                        "target_pairs": [{"output_name": o, "db_column": c} for o, c in target_pairs],
                        "visible_columns": visible_cols,
                        "augmented_sql": augmented_sql,
                        "ground_truth_table_csv": str(ground_truth_csv),
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
                        "sql_query": sql_query,
                        "ground_truth_column": ground_truth_text,
                        "target_pairs": [{"output_name": o, "db_column": c} for o, c in target_pairs],
                        "status": status,
                        "error": error,
                    },
                )

            manifest_rows.append(
                {
                    "item_id": item_id,
                    "csv_row_number": csv_row_number,
                    "question": question,
                    "ground_truth_column": ground_truth_text,
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
        ["item_id", "csv_row_number", "question", "ground_truth_column", "status", "error", "question_dir", "ground_truth_table_csv"],
        manifest_rows,
    )
    write_json(
        summary_path,
        {
            "input_csv": str(csv_path),
            "db_path": str(db_path),
            "output_dir": str(output_dir),
            "question_count": len(manifest_rows),
            "exported_count": exported_count,
            "skipped_count": skipped_count,
            "manifest_csv": str(manifest_path),
        },
    )
    print(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_dir": str(output_dir),
                "question_count": len(manifest_rows),
                "exported_count": exported_count,
                "skipped_count": skipped_count,
                "manifest_csv": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
