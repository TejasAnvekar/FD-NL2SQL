#!/usr/bin/env python3
"""Split the annotated CSV into Table and Retrieval subsets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


LABEL_HEADER = "Table or Retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split rows into Table and Retrieval CSVs based on the 'Table or Retrieval' column."
    )
    parser.add_argument(
        "--csv_path",
        default="/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--table_output",
        default="",
        help="Optional output CSV for all rows whose label starts with 'Table'.",
    )
    parser.add_argument(
        "--retrieval_output",
        default="",
        help="Optional output CSV for rows labeled 'Retrieval'.",
    )
    parser.add_argument(
        "--summary_output",
        default="",
        help="Optional summary JSON path.",
    )
    return parser.parse_args()


def default_output_paths(csv_path: Path) -> Dict[str, Path]:
    base = csv_path.with_suffix("")
    return {
        "table": base.with_name(base.name + "_table_rows.csv"),
        "retrieval": base.with_name(base.name + "_retrieval_rows.csv"),
        "summary": base.with_name(base.name + "_table_retrieval_split.summary.json"),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_label(label: str) -> str:
    text = (label or "").strip()
    if text.startswith("Table"):
        return "table"
    if text == "Retrieval":
        return "retrieval"
    return "other"


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    defaults = default_output_paths(csv_path)
    table_output = Path(args.table_output).expanduser().resolve() if args.table_output else defaults["table"]
    retrieval_output = Path(args.retrieval_output).expanduser().resolve() if args.retrieval_output else defaults["retrieval"]
    summary_output = Path(args.summary_output).expanduser().resolve() if args.summary_output else defaults["summary"]

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if LABEL_HEADER not in fieldnames:
        raise ValueError(f"CSV must contain {LABEL_HEADER!r}.")

    table_rows: List[Dict[str, str]] = []
    retrieval_rows: List[Dict[str, str]] = []
    other_rows: List[Dict[str, str]] = []
    raw_counts: Counter[str] = Counter()

    for row in rows:
        label = row.get(LABEL_HEADER, "")
        raw_counts[label] += 1
        bucket = classify_label(label)
        if bucket == "table":
            table_rows.append(row)
        elif bucket == "retrieval":
            retrieval_rows.append(row)
        else:
            other_rows.append(row)

    write_csv(table_output, fieldnames, table_rows)
    write_csv(retrieval_output, fieldnames, retrieval_rows)

    summary = {
        "input_csv": str(csv_path),
        "table_output_csv": str(table_output),
        "retrieval_output_csv": str(retrieval_output),
        "row_count": len(rows),
        "table_count": len(table_rows),
        "retrieval_count": len(retrieval_rows),
        "other_count": len(other_rows),
        "raw_label_counts": dict(sorted(raw_counts.items())),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
