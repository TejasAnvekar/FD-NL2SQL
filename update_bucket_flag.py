#!/usr/bin/env python3
"""Add or update a bucket flag for follow-up-month bucketing/classification questions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence


QUESTION_HEADER = "natural_language_query"
BUCKET_HEADER = "bucket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark questions that bucket or classify based on follow-up months."
    )
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        required=True,
        help="One or more CSV files to update in place.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("â€“", "-").replace("–", "-").replace("—", "-")
    normalized = normalized.replace("followup", "follow-up")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def is_followup_month_bucket_question(question: str) -> bool:
    q = normalize_text(question)
    has_followup = "follow-up" in q or "follow up" in q
    has_month = "month" in q or "months" in q
    if not (has_followup and has_month):
        return False

    has_bucket = "bucket" in q
    has_classify = "classify" in q and ("maturity" in q or ">=" in q or "≥" in q or "immature" in q or "mature" in q)
    return has_bucket or has_classify


def write_csv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_csv(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if QUESTION_HEADER not in fieldnames:
        raise ValueError(f"{path} does not contain {QUESTION_HEADER!r}.")
    if BUCKET_HEADER not in fieldnames:
        fieldnames.append(BUCKET_HEADER)

    yes_count = 0
    for row in rows:
        is_bucket = is_followup_month_bucket_question(row.get(QUESTION_HEADER, ""))
        row[BUCKET_HEADER] = "yes" if is_bucket else ""
        yes_count += int(is_bucket)

    write_csv(path, fieldnames, rows)
    return {
        "csv_path": str(path),
        "row_count": len(rows),
        "bucket_yes_count": yes_count,
        "bucket_blank_count": len(rows) - yes_count,
    }


def main() -> int:
    args = parse_args()
    summaries = [process_csv(Path(raw_path).expanduser().resolve()) for raw_path in args.csv_paths]
    print(json.dumps({"files": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
