#!/usr/bin/env python3
"""Build a small CSV with one eligible question per final-column category."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_hidden_column_sql_eval import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, fetch_schema, load_csv_rows  # noqa: E402
from utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select one eligible question per final_column from the annotated CSV."
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--column_used_key", default="column_used")
    ap.add_argument("--final_column_key", default="final_column")
    ap.add_argument("--require_distinct_source_target", type=int, default=1)
    ap.add_argument("--max_per_group", type=int, default=1)
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--output_summary_json", default="")
    return ap.parse_args()


def default_output_path(csv_path: Path, max_per_group: int) -> Path:
    stem = csv_path.stem
    suffix = f"_sample_{max_per_group}_per_final_column.csv"
    return csv_path.with_name(stem + suffix)


def filter_rows(
    rows: Sequence[Dict[str, str]],
    *,
    schema_cols: Sequence[str],
    question_key: str,
    column_used_key: str,
    final_column_key: str,
    require_distinct_source_target: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    schema_set = set(schema_cols)
    out: List[Dict[str, Any]] = []
    skipped = Counter()

    for csv_row_number, row in enumerate(rows, start=2):
        question = (row.get(question_key) or "").strip()
        source_column = (row.get(column_used_key) or "").strip()
        final_column = (row.get(final_column_key) or "").strip()

        if not final_column or final_column == "no_match" or final_column not in schema_set:
            skipped["final_column_not_usable"] += 1
            continue
        if not source_column or source_column not in schema_set:
            skipped["column_used_not_usable"] += 1
            continue
        if require_distinct_source_target and source_column == final_column:
            skipped["source_equals_hidden"] += 1
            continue
        if not question:
            skipped["missing_question"] += 1
            continue

        item = dict(row)
        item["_csv_row_number"] = csv_row_number
        out.append(item)

    return out, dict(skipped)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_path(csv_path, args.max_per_group)
    output_summary_json = (
        Path(args.output_summary_json).expanduser().resolve()
        if args.output_summary_json
        else output_csv.with_suffix(".summary.json")
    )

    conn = sqlite3.connect(db_path)
    try:
        schema_cols = fetch_schema(conn, args.table_name)
    finally:
        conn.close()

    rows = load_csv_rows(csv_path)
    eligible_rows, skipped = filter_rows(
        rows,
        schema_cols=schema_cols,
        question_key=args.question_key,
        column_used_key=args.column_used_key,
        final_column_key=args.final_column_key,
        require_distinct_source_target=bool(args.require_distinct_source_target),
    )

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        by_group[str(row[args.final_column_key])].append(row)

    selected: List[Dict[str, Any]] = []
    for final_column in sorted(by_group):
        for row in by_group[final_column][: max(1, args.max_per_group)]:
            selected.append(row)

    write_csv(output_csv, selected)

    summary = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "table_name": args.table_name,
        "max_per_group": args.max_per_group,
        "selected_count": len(selected),
        "eligible_count": len(eligible_rows),
        "skipped": skipped,
        "groups_available": {group: len(items) for group, items in sorted(by_group.items())},
        "groups_selected": {
            group: min(len(items), max(1, args.max_per_group))
            for group, items in sorted(by_group.items())
        },
        "output_csv": str(output_csv),
    }
    write_json(output_summary_json, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
