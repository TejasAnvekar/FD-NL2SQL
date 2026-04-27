#!/usr/bin/env python3
"""Evaluate predicted tables against ground-truth tables using tabular metrics."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval_run_baselines_v2 import (
    align_columns,
    bertscore_f1_avg,
    canonical_value,
    chrf_corpus,
    hungarian_min_cost_square,
    multiset_rows,
    project_rows,
    row_similarity,
    row_to_text,
    rouge_l_f1_avg,
    safe_div,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate per-question predicted tables against ground-truth tables "
            "using the tabular metrics from eval_run_baselines_v2.py."
        )
    )
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--question_subdir", default="questions")
    ap.add_argument("--pred_filename", default="formatted_predictions_like_ground_truth.csv")
    ap.add_argument("--gt_filename", default="ground_truth_table.csv")
    ap.add_argument("--metadata_filename", default="metadata.json")
    ap.add_argument("--skip_columns", nargs="*", default=["row_index"])
    ap.add_argument("--compute_bertscore", type=int, default=0)
    ap.add_argument("--comparison_csv", default="")
    return ap.parse_args()


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        vals.append(float(value))
    return float(sum(vals) / len(vals)) if vals else 0.0


def read_csv_table(path: Path) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Tuple[Any, ...]] = []
        for row in reader:
            rows.append(tuple(parse_cell(row.get(col, "")) for col in fieldnames))
    return fieldnames, rows


def parse_cell(value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    low = text.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if low.isdigit() or (low.startswith("-") and low[1:].isdigit()):
            return int(low)
    except Exception:
        pass
    try:
        return float(text)
    except Exception:
        return text


def filter_columns(
    cols: List[str], rows: List[Tuple[Any, ...]], skip_columns: Sequence[str]
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    skip = set(skip_columns or [])
    keep_indices = [idx for idx, col in enumerate(cols) if col not in skip]
    keep_cols = [cols[idx] for idx in keep_indices]
    keep_rows = [tuple(row[idx] for idx in keep_indices) for row in rows]
    return keep_cols, keep_rows


def compare_tables(
    pred_cols: List[str],
    pred_rows: List[Tuple[Any, ...]],
    gt_cols: List[str],
    gt_rows: List[Tuple[Any, ...]],
    *,
    compute_bertscore: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred_row_count": len(pred_rows),
        "gt_row_count": len(gt_rows),
        "pred_col_count": len(pred_cols),
        "gt_col_count": len(gt_cols),
        "column_alignment_score": 0.0,
        "matched_row_pairs": 0,
        "hard_overlap_rows": 0,
        "soft_overlap_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "row_jaccard": 0.0,
        "normalization_factor": 0.0,
        "chrf": None,
        "rouge_l_f1": None,
        "bertscore_f1": None,
        "exact_table_match": False,
        "error": None,
    }

    if not gt_cols:
        out["error"] = "EMPTY_GT_COLUMNS"
        return out
    if not pred_cols:
        out["error"] = "EMPTY_PRED_COLUMNS"
        return out

    col_pairs, col_score = align_columns(pred_cols, gt_cols, pred_rows, gt_rows)
    out["column_alignment_score"] = float(col_score)
    if not col_pairs:
        out["error"] = "NO_COLUMN_ALIGNMENT"
        return out

    pred_idx = [i for i, _, _ in col_pairs]
    gt_idx = [j for _, j, _ in col_pairs]
    pred_proj = project_rows(pred_rows, pred_idx)
    gt_proj = project_rows(gt_rows, gt_idx)

    out["exact_table_match"] = (multiset_rows(pred_proj) == multiset_rows(gt_proj))

    m = len(pred_proj)
    n = len(gt_proj)
    k = max(m, n)
    if k == 0:
        out.update(
            {
                "matched_row_pairs": 0,
                "hard_overlap_rows": 0,
                "soft_overlap_score": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "row_jaccard": 1.0,
                "normalization_factor": 1.0,
                "chrf": 100.0,
                "rouge_l_f1": 1.0,
                "bertscore_f1": 1.0 if compute_bertscore else None,
                "exact_table_match": True,
            }
        )
        return out

    sim = [[0.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = row_similarity(pred_proj[i], gt_proj[j])
    cost = [[1.0 - sim[i][j] for j in range(k)] for i in range(k)]
    assignment, _ = hungarian_min_cost_square(cost)

    hard_overlap = 0
    soft_overlap = 0.0
    matched_pred_texts: List[str] = []
    matched_gt_texts: List[str] = []
    for i in range(m):
        j = assignment[i]
        if 0 <= j < n:
            s = sim[i][j]
            soft_overlap += s
            if s >= 1.0 - 1e-12:
                hard_overlap += 1
            matched_pred_texts.append(row_to_text(pred_proj[i]))
            matched_gt_texts.append(row_to_text(gt_proj[j]))

    p = safe_div(soft_overlap, m) if m > 0 else (1.0 if n == 0 else 0.0)
    r = safe_div(soft_overlap, n) if n > 0 else (1.0 if m == 0 else 0.0)
    f1 = safe_div(2.0 * p * r, p + r) if (p + r) > 0 else 0.0
    denom = (m + n - soft_overlap)
    jacc = safe_div(soft_overlap, denom) if denom > 0 else 1.0

    out["matched_row_pairs"] = len(matched_pred_texts)
    out["hard_overlap_rows"] = int(hard_overlap)
    out["soft_overlap_score"] = float(soft_overlap)
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)
    out["row_jaccard"] = float(jacc)
    out["normalization_factor"] = safe_div(float(hard_overlap), float(max(1, m)))
    out["chrf"] = chrf_corpus(matched_pred_texts, matched_gt_texts)
    out["rouge_l_f1"] = rouge_l_f1_avg(matched_pred_texts, matched_gt_texts)
    out["bertscore_f1"] = bertscore_f1_avg(matched_pred_texts, matched_gt_texts, enabled=compute_bertscore)
    return out


def read_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def evaluate_run_dir(
    run_dir: Path,
    *,
    question_subdir: str,
    pred_filename: str,
    gt_filename: str,
    metadata_filename: str,
    skip_columns: Sequence[str],
    compute_bertscore: bool,
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

        gt_cols, gt_rows = read_csv_table(gt_path)
        pred_cols, pred_rows = read_csv_table(pred_path)
        gt_cols, gt_rows = filter_columns(gt_cols, gt_rows, skip_columns)
        pred_cols, pred_rows = filter_columns(pred_cols, pred_rows, skip_columns)
        metrics = compare_tables(
            pred_cols,
            pred_rows,
            gt_cols,
            gt_rows,
            compute_bertscore=compute_bertscore,
        )
        counters["evaluated_count"] += 1
        question_rows.append(
            {
                **row_base,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "pred_cols_json": json.dumps(pred_cols, ensure_ascii=False),
                "gt_cols_json": json.dumps(gt_cols, ensure_ascii=False),
                **metrics,
            }
        )

    exact_match_values = [
        1.0 if bool(row.get("exact_table_match")) else 0.0
        for row in question_rows
        if row.get("error") in (None, "", "EMPTY_PRED_COLUMNS")
    ]

    summary = {
        "run_dir": str(run_dir),
        **counters,
        "avg_pred_row_count": avg_numeric(question_rows, "pred_row_count"),
        "avg_gt_row_count": avg_numeric(question_rows, "gt_row_count"),
        "avg_column_alignment_score": avg_numeric(question_rows, "column_alignment_score"),
        "avg_precision": avg_numeric(question_rows, "precision"),
        "avg_recall": avg_numeric(question_rows, "recall"),
        "avg_f1": avg_numeric(question_rows, "f1"),
        "avg_row_jaccard": avg_numeric(question_rows, "row_jaccard"),
        "avg_chrf": avg_numeric(question_rows, "chrf"),
        "avg_rouge_l_f1": avg_numeric(question_rows, "rouge_l_f1"),
        "avg_bertscore_f1": avg_numeric(question_rows, "bertscore_f1"),
        "exact_table_match_rate": mean(exact_match_values) if exact_match_values else 0.0,
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
            skip_columns=args.skip_columns,
            compute_bertscore=bool(int(args.compute_bertscore)),
        )

        out_dir = run_dir / "tabular_eval_v3"
        question_csv = out_dir / "question_metrics.csv"
        summary_json = out_dir / "summary.json"
        write_csv(question_csv, result["question_rows"])
        write_json(summary_json, result["summary"])

        comparison_rows.append(result["summary"])
        print(f"{run_dir}\tevaluated={result['summary']['evaluated_count']}\tf1={result['summary']['avg_f1']:.4f}\texact={result['summary']['exact_table_match_rate']:.4f}")

    comparison_csv = args.comparison_csv
    if not comparison_csv and len(comparison_rows) > 1:
        comparison_csv = str(run_dirs[0].parent / "tabular_eval_v3_comparison.csv")
    if comparison_csv:
        write_csv(Path(comparison_csv).expanduser().resolve(), comparison_rows)


if __name__ == "__main__":
    main()
