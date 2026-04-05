#!/usr/bin/env python3
"""Update the 'Table or Retrieval' column based on match-score heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KEY_HEADER = "expected_llm_response_key_column_match"
VALUE_HEADER = "expected_llm_response_value_column_candidates"
GROUND_TRUTH_HEADER = "ground_truth_column"
OUTPUT_HEADER = "Table or Retrieval"

KEY_MATCH_RE = re.compile(r"->\s*(?P<column>.+?)\s*\[MATCH\s+(?P<score>\d+(?:\.\d+)?)\]")
KEY_BEST_RE = re.compile(r"best:\s*(?P<column>.+?),\s*(?P<score>\d+(?:\.\d+)?)\)")
VALUE_SCORE_RE = re.compile(r"(?P<column>.+?)\s*\[(?P<score>\d+(?:\.\d+)?)\]")
VALUE_BEST_RE = re.compile(r"best:\s*(?P<column>.+?),\s*(?P<score>\d+(?:\.\d+)?)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the 'Table or Retrieval' CSV column using key/value match scores."
    )
    parser.add_argument(
        "--csv_path",
        default="/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv",
        help="Input CSV path to update.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. Defaults to updating the input file in place.",
    )
    return parser.parse_args()


def parse_key_best(text: str) -> Tuple[Optional[str], float]:
    text = (text or "").strip()
    if not text:
        return None, 0.0

    match = KEY_MATCH_RE.search(text)
    if match:
        return match.group("column").strip(), float(match.group("score"))

    best = KEY_BEST_RE.search(text)
    if best:
        return best.group("column").strip(), float(best.group("score"))

    return None, 0.0


def parse_value_best(text: str) -> Tuple[Optional[str], float, List[str]]:
    text = (text or "").strip()
    if not text:
        return None, 0.0, []

    rhs = text.split("->", 1)[1].strip() if "->" in text else text
    scored: List[Tuple[str, float]] = []
    exact_columns: List[str] = []
    for match in VALUE_SCORE_RE.finditer(rhs):
        column = match.group("column").strip().rstrip(",")
        score = float(match.group("score"))
        scored.append((column, score))
        if abs(score - 1.0) < 1e-9:
            exact_columns.append(column)

    if scored:
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[0][0], scored[0][1], exact_columns

    best = VALUE_BEST_RE.search(text)
    if best:
        return best.group("column").strip(), float(best.group("score")), []

    return None, 0.0, []


def decide_label(row: Dict[str, str]) -> str:
    ground_truth = (row.get(GROUND_TRUTH_HEADER, "") or "").strip()
    if not ground_truth or ground_truth == "no_match" or "no_match" in ground_truth:
        return "Retrieval"
    if " | " in ground_truth or " -> " in ground_truth:
        return "Table"
    return f"Table ({ground_truth})"


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else csv_path

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if GROUND_TRUTH_HEADER not in fieldnames:
        raise ValueError(f"CSV must contain {GROUND_TRUTH_HEADER!r}.")

    if OUTPUT_HEADER not in fieldnames:
        fieldnames.append(OUTPUT_HEADER)

    counts: Dict[str, int] = {}
    for row in rows:
        label = decide_label(row)
        row[OUTPUT_HEADER] = label
        counts[label] = counts.get(label, 0) + 1

    write_csv(output_path, rows, fieldnames)
    print(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_csv": str(output_path),
                "row_count": len(rows),
                "label_counts": counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
