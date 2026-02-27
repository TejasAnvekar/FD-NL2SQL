#!/usr/bin/env python3
"""Compare staged GPT method batches against multiple GPT baseline eval JSON files.

Outputs:
- Per-baseline detailed comparison JSON
- One combined summary JSON
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from compare_fewshot_vs_staged_batches import (
    BOOL_METRICS,
    NUM_METRICS,
    compare_one,
    compare_overall,
    load_eval,
    to_item_map,
)


def _sanitize_label(text: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return label or "baseline"


def baseline_label_from_path(path: Path) -> str:
    stem = path.stem
    # GPT baseline eval files are all named eval_run_baselines_v2.json.
    # Use parent folder (zero_shot/few_shot/cot) as the comparator label.
    if stem == "eval_run_baselines_v2":
        stem = path.parent.name or stem
    return _sanitize_label(stem)


def default_batch_evals(root: Path) -> List[Path]:
    chatgpt_pipeline = root / "results" / "chatgpt_ready_pipeline"
    singles = sorted(chatgpt_pipeline.glob("single_*.eval.json"))
    if singles:
        return singles
    return [
        root / "method" / "batch_qwen3_30b" / "batch_1" / "batch_1.eval.json",
        root / "method" / "batch_qwen3_30b" / "batch_2" / "batch_2.eval.json",
        root / "method" / "batch_qwen3_30b" / "batch_3" / "batch_3.eval.json",
    ]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    default_baselines = [
        root / "results" / "gpt_predictions" / "cot" / "eval_run_baselines_v2.json",
        root / "results" / "gpt_predictions" / "few_shot" / "eval_run_baselines_v2.json",
        root / "results" / "gpt_predictions" / "zero_shot" / "eval_run_baselines_v2.json",
    ]

    ap = argparse.ArgumentParser(description="Evaluate method batches vs multiple GPT baseline eval JSON files.")
    ap.add_argument(
        "--baseline-evals",
        nargs="+",
        default=[str(p) for p in default_baselines],
        help="Baseline eval JSON paths (e.g., cot/few-shot/zero-shot).",
    )
    ap.add_argument(
        "--batch-evals",
        nargs="+",
        default=[str(p) for p in default_batch_evals(root)],
        help="Method batch eval JSON paths.",
    )
    ap.add_argument(
        "--id-source",
        choices=["pred_nonempty", "all"],
        default="pred_nonempty",
        help="Question set per batch: rows with non-empty pred_sql, or all rows.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(root / "results" / "gpt_method_vs_baselines"),
        help="Directory to save comparison reports.",
    )
    return ap.parse_args()


def relabel_result(result: Dict[str, Any], baseline_label: str) -> Dict[str, Any]:
    delta_key = f"delta_batch_minus_{baseline_label}"

    counts = result.get("counts")
    if isinstance(counts, dict) and "missing_in_fewshot" in counts:
        counts[f"missing_in_{baseline_label}"] = counts.pop("missing_in_fewshot")

    h2h = result.get("exact_head_to_head")
    if isinstance(h2h, dict) and "fewshot_only_true" in h2h:
        h2h[f"{baseline_label}_only_true"] = h2h.pop("fewshot_only_true")

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        for row in metrics.values():
            if not isinstance(row, dict):
                continue
            if "fewshot" in row:
                row[baseline_label] = row.pop("fewshot")
            if "delta_batch_minus_fewshot" in row:
                row[delta_key] = row.pop("delta_batch_minus_fewshot")

    result["baseline_label"] = baseline_label
    return result


def print_report_with_label(result: Dict[str, Any], baseline_label: str) -> None:
    delta_key = f"delta_batch_minus_{baseline_label}"
    missing_key = f"missing_in_{baseline_label}"
    baseline_only_key = f"{baseline_label}_only_true"

    print("=" * 90)
    print(f"Batch: {result['batch']}")
    print(f"ID source: {result['question_set_source']}")
    c = result["counts"]
    print(
        f"Scope IDs: {c['batch_scope_ids']} "
        f"(raw={c.get('batch_scope_ids_raw', c['batch_scope_ids'])}, deduped={c.get('deduped_scope_ids', 0)}) "
        f"| Common IDs: {c['common_ids']} | Missing in {baseline_label}: {c.get(missing_key, 0)}"
    )
    h = result["exact_head_to_head"]
    print(
        "Exact head-to-head: "
        f"both={h['both_true']} batch_only={h['batch_only_true']} "
        f"{baseline_label}_only={h.get(baseline_only_key, 0)} neither={h['both_false']}"
    )
    print("-" * 90)
    for key in BOOL_METRICS + NUM_METRICS:
        m = result["metrics"][key]
        print(
            f"{key:24s} "
            f"batch={m['batch']:.6f} "
            f"{baseline_label}={m[baseline_label]:.6f} "
            f"delta={m[delta_key]:+.6f}"
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_objs: Dict[str, Dict[str, Any]] = {}
    batch_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for p in args.batch_evals:
        path = Path(p)
        obj = load_eval(path)
        batch_objs[path.stem] = obj
        batch_rows_by_name[path.stem] = obj["per_item"]

    summary_rows: List[Dict[str, Any]] = []

    for bpath in args.baseline_evals:
        baseline_path = Path(bpath)
        baseline_label = baseline_label_from_path(baseline_path)

        baseline_obj = load_eval(baseline_path)
        baseline_map = to_item_map(baseline_obj["per_item"])

        print("=" * 100)
        print(f"Baseline: {baseline_path.name} ({baseline_label})")

        per_batch_results: List[Dict[str, Any]] = []
        for mpath in args.batch_evals:
            mp = Path(mpath)
            obj = batch_objs[mp.stem]
            res = compare_one(
                batch_name=mp.stem,
                batch_rows=obj["per_item"],
                few_map=baseline_map,
                id_source=args.id_source,
            )
            relabel_result(res, baseline_label)
            per_batch_results.append(res)
            print_report_with_label(res, baseline_label)

        overall = compare_overall(
            batch_results=per_batch_results,
            batch_rows_by_name=batch_rows_by_name,
            few_map=baseline_map,
            id_source=args.id_source,
        )
        relabel_result(overall, baseline_label)
        print_report_with_label(overall, baseline_label)

        report = {
            "baseline_eval": str(baseline_path),
            "baseline_label": baseline_label,
            "batch_evals": [str(Path(x)) for x in args.batch_evals],
            "id_source": args.id_source,
            "results": per_batch_results,
            "overall": overall,
        }
        out_file = out_dir / f"{baseline_label}_vs_method_batches.json"
        out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        delta_key = f"delta_batch_minus_{baseline_label}"
        overall_metrics = overall.get("metrics", {})
        em_block = overall_metrics.get("exact_exec_match", {}) if isinstance(overall_metrics, dict) else {}
        f1_block = overall_metrics.get("f1", {}) if isinstance(overall_metrics, dict) else {}
        ast_block = overall_metrics.get("sql_ast_similarity", {}) if isinstance(overall_metrics, dict) else {}
        summary_rows.append(
            {
                "baseline": baseline_label,
                "baseline_eval": str(baseline_path),
                "common_ids": (overall.get("counts") or {}).get("common_ids"),
                "exact_exec_match_batch": em_block.get("batch"),
                "exact_exec_match_baseline": em_block.get(baseline_label),
                "exact_exec_match_delta": em_block.get(delta_key),
                "f1_batch": f1_block.get("batch"),
                "f1_baseline": f1_block.get(baseline_label),
                "f1_delta": f1_block.get(delta_key),
                "sql_ast_similarity_delta": ast_block.get(delta_key),
            }
        )

    summary = {
        "baseline_evals": [str(Path(x)) for x in args.baseline_evals],
        "batch_evals": [str(Path(x)) for x in args.batch_evals],
        "id_source": args.id_source,
        "summary_rows": summary_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Overall Summary (method - baseline)")
    for r in summary_rows:
        print(
            f"{r['baseline']}: "
            f"EM delta={float(r['exact_exec_match_delta'] or 0.0):+0.4f}, "
            f"F1 delta={float(r['f1_delta'] or 0.0):+0.4f}, "
            f"AST delta={float(r['sql_ast_similarity_delta'] or 0.0):+0.4f}"
        )
    print(f"Saved reports to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
