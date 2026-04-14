#!/usr/bin/env python3
"""Evaluate row-aligned derived-value predictions against ground-truth tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


IDENTIFIER_COLUMNS = ["NCT", "PubMed ID", "Trial name", "source_value"]
META_COLUMNS = {"row_index", "derived_expected_llm_response"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate predicted derived values against ground-truth tables by "
            "aligning rows on visible identifiers and scoring the hidden target columns."
        )
    )
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--question_subdir", default="questions")
    ap.add_argument("--pred_filename", default="formatted_predictions_like_ground_truth.csv")
    ap.add_argument("--gt_filename", default="ground_truth_table.csv")
    ap.add_argument("--metadata_filename", default="metadata.json")
    ap.add_argument("--comparison_csv", default="")
    return ap.parse_args()


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        vals.append(float(value))
    return float(sum(vals) / len(vals)) if vals else 0.0


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
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


def canonical_atomic(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
            return int(round(value))
        return round(value, 6)
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d+", text):
            num = float(text)
            if math.isfinite(num) and math.isclose(num, round(num), rel_tol=0.0, abs_tol=1e-12):
                return int(round(num))
            return round(num, 6)
    except Exception:
        pass
    return text


def normalized_string(value: Any) -> str:
    atom = canonical_atomic(value)
    if atom is None:
        return ""
    if isinstance(atom, bool):
        return "true" if atom else "false"
    if isinstance(atom, (int, float)):
        return str(atom)
    text = str(atom).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def aggressive_normalized_string(value: Any) -> str:
    text = normalized_string(value)
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text)


def values_equal_exact(a: Any, b: Any) -> bool:
    return canonical_atomic(a) == canonical_atomic(b)


def values_equal_normalized(a: Any, b: Any) -> bool:
    return aggressive_normalized_string(a) == aggressive_normalized_string(b)


def payload_to_target_map(payload: Any, target_columns: Sequence[str]) -> Dict[str, Any]:
    if len(target_columns) == 1 and not isinstance(payload, dict):
        return {target_columns[0]: payload}
    if isinstance(payload, dict):
        return {col: payload.get(col) for col in target_columns}
    return {}


def target_columns_for_rows(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    return [col for col in cols if col not in META_COLUMNS and col not in IDENTIFIER_COLUMNS]


def row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return tuple(str(row.get(col, "") or "") for col in IDENTIFIER_COLUMNS)


def build_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    return {row_key(row): row for row in rows}


def score_aligned_row(
    pred_row: Optional[Dict[str, Any]],
    gt_row: Dict[str, Any],
    target_columns: Sequence[str],
) -> Dict[str, Any]:
    gt_payload = payload_to_target_map(parse_jsonish(gt_row.get("derived_expected_llm_response")), target_columns)
    pred_payload = payload_to_target_map(
        parse_jsonish(pred_row.get("derived_expected_llm_response")) if pred_row else None,
        target_columns,
    )

    exact_targets = 0
    normalized_targets = 0
    for col in target_columns:
        pred_value = pred_row.get(col) if pred_row else None
        gt_value = gt_row.get(col)
        if values_equal_exact(pred_value, gt_value):
            exact_targets += 1
        if values_equal_normalized(pred_value, gt_value):
            normalized_targets += 1

    target_count = len(target_columns)
    target_all_exact = bool(target_count) and exact_targets == target_count
    target_all_normalized = bool(target_count) and normalized_targets == target_count

    payload_all_exact = all(
        values_equal_exact(pred_payload.get(col), gt_payload.get(col)) for col in target_columns
    ) if target_columns else False
    payload_all_normalized = all(
        values_equal_normalized(pred_payload.get(col), gt_payload.get(col)) for col in target_columns
    ) if target_columns else False

    return {
        "target_column_count": target_count,
        "target_exact_cell_count": exact_targets,
        "target_normalized_cell_count": normalized_targets,
        "target_cell_exact_accuracy": safe_div(float(exact_targets), float(target_count)),
        "target_cell_normalized_accuracy": safe_div(float(normalized_targets), float(target_count)),
        "target_all_exact": target_all_exact,
        "target_all_normalized": target_all_normalized,
        "payload_all_exact": payload_all_exact,
        "payload_all_normalized": payload_all_normalized,
        "gt_payload_json": json.dumps(gt_payload, ensure_ascii=False, sort_keys=True),
        "pred_payload_json": json.dumps(pred_payload, ensure_ascii=False, sort_keys=True),
    }


def evaluate_question(
    pred_rows: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    target_columns = target_columns_for_rows(gt_rows)
    pred_index = build_index(pred_rows)
    gt_index = build_index(gt_rows)

    gt_count = len(gt_rows)
    pred_count = len(pred_rows)
    matched_key_count = 0
    missing_pred_rows = 0
    extra_pred_rows = max(0, pred_count - len(set(pred_index.keys()) & set(gt_index.keys())))

    target_all_exact_count = 0
    target_all_normalized_count = 0
    payload_all_exact_count = 0
    payload_all_normalized_count = 0
    target_exact_cell_total = 0
    target_normalized_cell_total = 0
    target_cell_denominator_total = 0

    for gt_row in gt_rows:
        key = row_key(gt_row)
        pred_row = pred_index.get(key)
        if pred_row is None:
            missing_pred_rows += 1
            target_cell_denominator_total += len(target_columns)
            continue

        matched_key_count += 1
        row_scores = score_aligned_row(pred_row, gt_row, target_columns)
        target_exact_cell_total += row_scores["target_exact_cell_count"]
        target_normalized_cell_total += row_scores["target_normalized_cell_count"]
        target_cell_denominator_total += row_scores["target_column_count"]
        target_all_exact_count += 1 if row_scores["target_all_exact"] else 0
        target_all_normalized_count += 1 if row_scores["target_all_normalized"] else 0
        payload_all_exact_count += 1 if row_scores["payload_all_exact"] else 0
        payload_all_normalized_count += 1 if row_scores["payload_all_normalized"] else 0

    p_exact = safe_div(float(target_all_exact_count), float(pred_count))
    r_exact = safe_div(float(target_all_exact_count), float(gt_count))
    f1_exact = safe_div(2.0 * p_exact * r_exact, p_exact + r_exact) if (p_exact + r_exact) else 0.0

    p_norm = safe_div(float(target_all_normalized_count), float(pred_count))
    r_norm = safe_div(float(target_all_normalized_count), float(gt_count))
    f1_norm = safe_div(2.0 * p_norm * r_norm, p_norm + r_norm) if (p_norm + r_norm) else 0.0

    p_payload = safe_div(float(payload_all_exact_count), float(pred_count))
    r_payload = safe_div(float(payload_all_exact_count), float(gt_count))
    f1_payload = safe_div(2.0 * p_payload * r_payload, p_payload + r_payload) if (p_payload + r_payload) else 0.0

    p_payload_norm = safe_div(float(payload_all_normalized_count), float(pred_count))
    r_payload_norm = safe_div(float(payload_all_normalized_count), float(gt_count))
    f1_payload_norm = (
        safe_div(2.0 * p_payload_norm * r_payload_norm, p_payload_norm + r_payload_norm)
        if (p_payload_norm + r_payload_norm)
        else 0.0
    )

    return {
        "gt_row_count": gt_count,
        "pred_row_count": pred_count,
        "matched_key_row_count": matched_key_count,
        "missing_pred_rows": missing_pred_rows,
        "extra_pred_rows": extra_pred_rows,
        "target_columns_json": json.dumps(target_columns, ensure_ascii=False),
        "target_exact_cell_total": target_exact_cell_total,
        "target_normalized_cell_total": target_normalized_cell_total,
        "target_cell_denominator_total": target_cell_denominator_total,
        "target_cell_exact_accuracy": safe_div(
            float(target_exact_cell_total), float(target_cell_denominator_total)
        ),
        "target_cell_normalized_accuracy": safe_div(
            float(target_normalized_cell_total), float(target_cell_denominator_total)
        ),
        "target_all_exact_count": target_all_exact_count,
        "target_all_normalized_count": target_all_normalized_count,
        "payload_all_exact_count": payload_all_exact_count,
        "payload_all_normalized_count": payload_all_normalized_count,
        "target_row_exact_precision": p_exact,
        "target_row_exact_recall": r_exact,
        "target_row_exact_f1": f1_exact,
        "target_row_normalized_precision": p_norm,
        "target_row_normalized_recall": r_norm,
        "target_row_normalized_f1": f1_norm,
        "payload_exact_precision": p_payload,
        "payload_exact_recall": r_payload,
        "payload_exact_f1": f1_payload,
        "payload_normalized_precision": p_payload_norm,
        "payload_normalized_recall": r_payload_norm,
        "payload_normalized_f1": f1_payload_norm,
        "all_rows_target_exact": bool(gt_count > 0 and target_all_exact_count == gt_count and pred_count == gt_count),
        "all_rows_target_normalized": bool(
            gt_count > 0 and target_all_normalized_count == gt_count and pred_count == gt_count
        ),
        "all_rows_payload_exact": bool(gt_count > 0 and payload_all_exact_count == gt_count and pred_count == gt_count),
        "all_rows_payload_normalized": bool(
            gt_count > 0 and payload_all_normalized_count == gt_count and pred_count == gt_count
        ),
    }


def evaluate_run_dir(
    run_dir: Path,
    *,
    question_subdir: str,
    pred_filename: str,
    gt_filename: str,
    metadata_filename: str,
) -> Dict[str, Any]:
    questions_dir = run_dir / question_subdir
    question_rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {
        "question_dir_count": 0,
        "evaluated_count": 0,
        "missing_pred_count": 0,
        "missing_gt_count": 0,
    }

    for question_dir in sorted(p for p in questions_dir.iterdir() if p.is_dir()):
        counters["question_dir_count"] += 1
        pred_path = question_dir / pred_filename
        gt_path = question_dir / gt_filename
        metadata = read_metadata(question_dir / metadata_filename)

        row_base = {
            "question_dir": str(question_dir),
            "item_id": metadata.get("item_id", question_dir.name),
            "csv_row_number": metadata.get("csv_row_number"),
            "question": metadata.get("question", ""),
            "column_used": metadata.get("column_used", ""),
        }

        if not gt_path.exists():
            counters["missing_gt_count"] += 1
            question_rows.append({**row_base, "error": "missing_gt_table"})
            continue
        if not pred_path.exists():
            counters["missing_pred_count"] += 1
            question_rows.append({**row_base, "error": "missing_pred_table"})
            continue

        gt_rows = read_csv_rows(gt_path)
        pred_rows = read_csv_rows(pred_path)
        metrics = evaluate_question(pred_rows, gt_rows)
        counters["evaluated_count"] += 1
        question_rows.append(
            {
                **row_base,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                **metrics,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        **counters,
        "avg_gt_row_count": avg_numeric(question_rows, "gt_row_count"),
        "avg_pred_row_count": avg_numeric(question_rows, "pred_row_count"),
        "avg_matched_key_row_count": avg_numeric(question_rows, "matched_key_row_count"),
        "avg_target_cell_exact_accuracy": avg_numeric(question_rows, "target_cell_exact_accuracy"),
        "avg_target_cell_normalized_accuracy": avg_numeric(question_rows, "target_cell_normalized_accuracy"),
        "avg_target_row_exact_precision": avg_numeric(question_rows, "target_row_exact_precision"),
        "avg_target_row_exact_recall": avg_numeric(question_rows, "target_row_exact_recall"),
        "avg_target_row_exact_f1": avg_numeric(question_rows, "target_row_exact_f1"),
        "avg_target_row_normalized_precision": avg_numeric(question_rows, "target_row_normalized_precision"),
        "avg_target_row_normalized_recall": avg_numeric(question_rows, "target_row_normalized_recall"),
        "avg_target_row_normalized_f1": avg_numeric(question_rows, "target_row_normalized_f1"),
        "avg_payload_exact_precision": avg_numeric(question_rows, "payload_exact_precision"),
        "avg_payload_exact_recall": avg_numeric(question_rows, "payload_exact_recall"),
        "avg_payload_exact_f1": avg_numeric(question_rows, "payload_exact_f1"),
        "avg_payload_normalized_precision": avg_numeric(question_rows, "payload_normalized_precision"),
        "avg_payload_normalized_recall": avg_numeric(question_rows, "payload_normalized_recall"),
        "avg_payload_normalized_f1": avg_numeric(question_rows, "payload_normalized_f1"),
        "all_rows_target_exact_rate": mean(
            1.0 if bool(row.get("all_rows_target_exact")) else 0.0
            for row in question_rows
            if row.get("error") in (None, "")
        ) if question_rows else 0.0,
        "all_rows_target_normalized_rate": mean(
            1.0 if bool(row.get("all_rows_target_normalized")) else 0.0
            for row in question_rows
            if row.get("error") in (None, "")
        ) if question_rows else 0.0,
        "all_rows_payload_exact_rate": mean(
            1.0 if bool(row.get("all_rows_payload_exact")) else 0.0
            for row in question_rows
            if row.get("error") in (None, "")
        ) if question_rows else 0.0,
        "all_rows_payload_normalized_rate": mean(
            1.0 if bool(row.get("all_rows_payload_normalized")) else 0.0
            for row in question_rows
            if row.get("error") in (None, "")
        ) if question_rows else 0.0,
    }
    return {"summary": summary, "question_rows": question_rows}


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    comparison_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        result = evaluate_run_dir(
            run_dir,
            question_subdir=args.question_subdir,
            pred_filename=args.pred_filename,
            gt_filename=args.gt_filename,
            metadata_filename=args.metadata_filename,
        )
        out_dir = run_dir / "derived_eval_v1"
        question_csv = out_dir / "question_metrics.csv"
        summary_json = out_dir / "summary.json"
        write_csv(question_csv, result["question_rows"])
        write_json(summary_json, result["summary"])
        comparison_rows.append(result["summary"])
        print(
            f"{run_dir}\tevaluated={result['summary']['evaluated_count']}"
            f"\ttarget_f1={result['summary']['avg_target_row_exact_f1']:.4f}"
            f"\tpayload_f1={result['summary']['avg_payload_exact_f1']:.4f}"
            f"\tnorm_target_f1={result['summary']['avg_target_row_normalized_f1']:.4f}"
            f"\tnorm_payload_f1={result['summary']['avg_payload_normalized_f1']:.4f}"
        )

    comparison_csv = args.comparison_csv
    if not comparison_csv and len(comparison_rows) > 1:
        comparison_csv = str(run_dirs[0].parent / "derived_eval_v1_comparison.csv")
    if comparison_csv:
        write_csv(Path(comparison_csv).expanduser().resolve(), comparison_rows)


if __name__ == "__main__":
    main()
