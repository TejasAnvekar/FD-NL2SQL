#!/usr/bin/env python3
"""Batch-format naive LLM prediction CSVs to match ground-truth table schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from format_llm_predictions_like_ground_truth import (
    convert_rows,
    build_fieldnames,
    infer_target_columns,
    read_json,
    read_csv_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Format all per-question llm_predictions.csv files inside one or more run directories."
    )
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--output_name", default="formatted_predictions_like_ground_truth.csv")
    ap.add_argument("--require_predicted_answer_json", type=int, default=0)
    return ap.parse_args()


def process_run_dir(run_dir: Path, output_name: str, require_predicted_answer_json: bool) -> dict:
    questions_dir = run_dir / "questions"
    formatted_count = 0
    skipped_count = 0
    written_paths: List[str] = []

    if not questions_dir.exists():
        return {
            "run_dir": str(run_dir),
            "formatted_count": 0,
            "skipped_count": 0,
            "written_paths": [],
            "error": "missing questions dir",
        }

    for question_dir in sorted(p for p in questions_dir.iterdir() if p.is_dir()):
        llm_predictions_csv = question_dir / "llm_predictions.csv"
        ground_truth_csv = question_dir / "ground_truth_table.csv"
        metadata_json = question_dir / "metadata.json"
        if not llm_predictions_csv.exists() or not ground_truth_csv.exists():
            skipped_count += 1
            continue

        llm_rows = read_csv_rows(llm_predictions_csv)
        gt_rows = read_csv_rows(ground_truth_csv)
        metadata = read_json(metadata_json) if metadata_json.exists() else {}
        gt_fieldnames = list(gt_rows[0].keys()) if gt_rows else []
        target_columns = infer_target_columns(gt_fieldnames, metadata)
        fieldnames = build_fieldnames(gt_fieldnames, target_columns)
        formatted_rows = convert_rows(
            llm_rows,
            fieldnames,
            target_columns=target_columns,
            require_predicted_answer_json=require_predicted_answer_json,
        )
        output_csv = question_dir / output_name
        write_csv(output_csv, formatted_rows, fieldnames)
        formatted_count += 1
        written_paths.append(str(output_csv))

    return {
        "run_dir": str(run_dir),
        "formatted_count": formatted_count,
        "skipped_count": skipped_count,
        "written_paths": written_paths,
    }


def main() -> None:
    args = parse_args()
    summaries = []
    total_formatted = 0
    total_skipped = 0
    for run_dir_str in args.run_dirs:
        summary = process_run_dir(
            Path(run_dir_str).expanduser().resolve(),
            args.output_name,
            bool(args.require_predicted_answer_json),
        )
        summaries.append(summary)
        total_formatted += int(summary.get("formatted_count", 0))
        total_skipped += int(summary.get("skipped_count", 0))

    for summary in summaries:
        print(
            "{run_dir}\tformatted={formatted_count}\tskipped={skipped_count}".format(
                **summary
            )
        )
    print(f"TOTAL\tformatted={total_formatted}\tskipped={total_skipped}")


if __name__ == "__main__":
    main()
