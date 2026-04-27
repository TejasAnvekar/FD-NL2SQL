#!/usr/bin/env python3
"""Convert question-level naive-LLM prediction CSVs into ground-truth-like tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Format llm_predictions.csv to match ground_truth_table.csv schema."
    )
    ap.add_argument("--llm_predictions_csv", required=True)
    ap.add_argument("--ground_truth_csv", required=True)
    ap.add_argument("--metadata_json", default="")
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--require_predicted_answer_json", type=int, default=0)
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def canonical_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return ", ".join("" if item is None else str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_payload(predicted_value: Any, target_columns: List[str]) -> str:
    if len(target_columns) == 1:
        payload = {target_columns[0]: predicted_value}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if isinstance(predicted_value, dict):
        payload = {}
        for col in target_columns:
            if col in predicted_value:
                payload[col] = predicted_value[col]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(predicted_value, ensure_ascii=False, sort_keys=True)


def infer_target_columns(gt_fieldnames: List[str], metadata: Dict[str, Any]) -> List[str]:
    target_columns = [
        col for col in gt_fieldnames
        if col not in {"row_index", "NCT", "PubMed ID", "Trial name", "source_value", "derived_expected_llm_response"}
    ]
    if target_columns:
        return target_columns

    expected_keys = metadata.get("expected_keys") or []
    if isinstance(expected_keys, list):
        inferred = [str(x) for x in expected_keys if str(x).strip()]
        if inferred:
            return inferred

    hidden_columns = metadata.get("hidden_columns") or []
    if isinstance(hidden_columns, list):
        inferred = [str(x) for x in hidden_columns if str(x).strip()]
        if inferred:
            return inferred

    return []


def build_fieldnames(gt_fieldnames: List[str], target_columns: List[str]) -> List[str]:
    if gt_fieldnames:
        return list(gt_fieldnames)
    return ["row_index", "NCT", "PubMed ID", "Trial name", "source_value", *target_columns, "derived_expected_llm_response"]


def convert_rows(
    llm_rows: List[Dict[str, str]],
    gt_fieldnames: List[str],
    *,
    target_columns: List[str],
    require_predicted_answer_json: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in llm_rows:
        predicted_text = (row.get("predicted_answer_json") or "").strip()
        actual_text = (row.get("actual_payload_json") or "").strip()
        if require_predicted_answer_json:
            if not predicted_text:
                continue
        else:
            if not predicted_text and not actual_text:
                continue

        predicted = parse_jsonish(predicted_text)
        out_row: Dict[str, Any] = {
            "row_index": len(out) + 1,
            "NCT": row.get("NCT", ""),
            "PubMed ID": row.get("PubMed ID", ""),
            "Trial name": row.get("Trial name", ""),
            "source_value": row.get("source_value", ""),
        }

        if len(target_columns) == 1:
            col = target_columns[0]
            if isinstance(predicted, dict):
                out_row[col] = canonical_scalar(predicted.get(col, ""))
            else:
                out_row[col] = canonical_scalar(predicted)
        else:
            for col in target_columns:
                if isinstance(predicted, dict):
                    out_row[col] = canonical_scalar(predicted.get(col, ""))
                else:
                    out_row[col] = ""

        out_row["derived_expected_llm_response"] = build_payload(predicted, target_columns)
        out.append(out_row)
    return out


def main() -> None:
    args = parse_args()
    llm_predictions_csv = Path(args.llm_predictions_csv).expanduser().resolve()
    ground_truth_csv = Path(args.ground_truth_csv).expanduser().resolve()
    metadata_json = Path(args.metadata_json).expanduser().resolve() if args.metadata_json else None
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else llm_predictions_csv.with_name("formatted_predictions_like_ground_truth.csv")
    )

    llm_rows = read_csv_rows(llm_predictions_csv)
    gt_rows = read_csv_rows(ground_truth_csv)
    metadata = read_json(metadata_json) if metadata_json and metadata_json.exists() else {}
    gt_fieldnames = list(gt_rows[0].keys()) if gt_rows else []
    target_columns = infer_target_columns(gt_fieldnames, metadata)
    fieldnames = build_fieldnames(gt_fieldnames, target_columns)
    formatted_rows = convert_rows(
        llm_rows,
        fieldnames,
        target_columns=target_columns,
        require_predicted_answer_json=bool(args.require_predicted_answer_json),
    )
    write_csv(output_csv, formatted_rows, fieldnames)
    print(str(output_csv))


if __name__ == "__main__":
    main()
