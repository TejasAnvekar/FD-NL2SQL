#!/usr/bin/env python3
"""Organize full baseline runs into per-method/per-model study folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_hidden_column_sql_eval import sanitize_name  # noqa: E402
from utils import write_json  # noqa: E402

DEFAULT_COSINE_RUN = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_cosine_full"
DEFAULT_TAPAS_RUN = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/question_table_copy_tapas_full"
DEFAULT_OUT_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/baseline_study_full"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Organize baseline runs into study folders.")
    ap.add_argument("--cosine_run_dir", default=DEFAULT_COSINE_RUN)
    ap.add_argument("--tapas_run_dir", default=DEFAULT_TAPAS_RUN)
    ap.add_argument("--output_root", default=DEFAULT_OUT_ROOT)
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows_out: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    for row in rows_out:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_question_dir_map(questions_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not questions_root.exists():
        return out
    for folder in questions_root.iterdir():
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            continue
        item_id = str(metadata.get("item_id") or "").strip()
        if item_id:
            out[item_id] = folder
    return out


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def organize_cosine(cosine_run_dir: Path, output_root: Path) -> Dict[str, Any]:
    question_results = read_csv_rows(cosine_run_dir / "all_question_results.csv")
    row_predictions = read_csv_rows(cosine_run_dir / "row_level_predictions.csv")
    baseline_summary = read_csv_rows(cosine_run_dir / "baseline_summary.csv")
    sample_manifest = read_csv_rows(cosine_run_dir / "sample_manifest.csv")
    question_dir_map = build_question_dir_map(cosine_run_dir / "questions")

    cosine_root = output_root / "strict_table_copy_cosine"
    cosine_root.mkdir(parents=True, exist_ok=True)

    model_names = sorted({str(row.get("model_name") or "").strip() for row in question_results if str(row.get("model_name") or "").strip()})
    summary: Dict[str, Any] = {"source_run_dir": str(cosine_run_dir), "models": {}}

    for model_name in model_names:
        model_root = cosine_root / model_name
        ensure_clean_dir(model_root)

        model_qrows = [row for row in question_results if str(row.get("model_name") or "").strip() == model_name]
        model_rrows = [row for row in row_predictions if str(row.get("model_name") or "").strip() == model_name]
        model_srows = [row for row in baseline_summary if str(row.get("model_name") or "").strip() == model_name]
        item_ids = {str(row.get("item_id") or "") for row in model_qrows}
        model_manifest = [row for row in sample_manifest if str(row.get("item_id") or "") in item_ids]

        write_csv(model_root / "sample_manifest.csv", model_manifest)
        write_csv(model_root / "all_question_results.csv", model_qrows)
        write_csv(model_root / "row_level_predictions.csv", model_rrows)
        write_csv(model_root / "baseline_summary.csv", model_srows)

        questions_root = model_root / "questions"
        questions_root.mkdir(parents=True, exist_ok=True)
        copied_questions = 0
        for item_id in sorted(item_ids):
            src_dir = question_dir_map.get(item_id)
            if not src_dir:
                continue
            dst_dir = questions_root / src_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            copy_if_exists(src_dir / "table_copy.csv", dst_dir / "table_copy.csv")
            copy_if_exists(src_dir / "ground_truth_table.csv", dst_dir / "ground_truth_table.csv")
            copy_if_exists(src_dir / "metadata.json", dst_dir / "metadata.json")
            topk_path = src_dir / "topk_retrievals.csv"
            if topk_path.exists():
                topk_rows = [row for row in read_csv_rows(topk_path) if str(row.get("model_name") or "").strip() == model_name]
                write_csv(dst_dir / "topk_retrievals.csv", topk_rows)
            copied_questions += 1

        model_summary = {
            "source_run_dir": str(cosine_run_dir),
            "model_name": model_name,
            "question_result_rows": len(model_qrows),
            "row_prediction_rows": len(model_rrows),
            "question_folder_count": copied_questions,
        }
        write_json(model_root / "summary.json", model_summary)
        summary["models"][model_name] = model_summary

    return summary


def organize_tapas(tapas_run_dir: Path, output_root: Path) -> Dict[str, Any]:
    tapas_root = output_root / "strict_table_copy_tapas" / "google_tapas_base_finetuned_wtq"
    ensure_clean_dir(tapas_root)

    copy_if_exists(tapas_run_dir / "sample_manifest.csv", tapas_root / "sample_manifest.csv")
    copy_if_exists(tapas_run_dir / "all_question_results.csv", tapas_root / "all_question_results.csv")
    copy_if_exists(tapas_run_dir / "baseline_summary.csv", tapas_root / "baseline_summary.csv")
    copy_if_exists(tapas_run_dir / "summary.json", tapas_root / "source_summary.json")

    src_questions = tapas_run_dir / "questions"
    dst_questions = tapas_root / "questions"
    copied_questions = 0
    if src_questions.exists():
        for folder in src_questions.iterdir():
            if not folder.is_dir():
                continue
            dst_dir = dst_questions / folder.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            copy_if_exists(folder / "table_copy.csv", dst_dir / "table_copy.csv")
            copy_if_exists(folder / "tapas_table_view.csv", dst_dir / "tapas_table_view.csv")
            copy_if_exists(folder / "ground_truth_table.csv", dst_dir / "ground_truth_table.csv")
            copy_if_exists(folder / "metadata.json", dst_dir / "metadata.json")
            copied_questions += 1

    summary = {
        "source_run_dir": str(tapas_run_dir),
        "model_name": "google_tapas_base_finetuned_wtq",
        "question_folder_count": copied_questions,
    }
    write_json(tapas_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()

    cosine_run_dir = Path(args.cosine_run_dir).expanduser().resolve()
    tapas_run_dir = Path(args.tapas_run_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    overall = {
        "cosine": organize_cosine(cosine_run_dir, output_root),
        "tapas": organize_tapas(tapas_run_dir, output_root),
    }
    write_json(output_root / "summary.json", overall)


if __name__ == "__main__":
    main()
