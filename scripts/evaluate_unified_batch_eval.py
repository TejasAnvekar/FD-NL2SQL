#!/usr/bin/env python3
"""Unified batch comparison for GPT/Gemma/Qwen baselines vs method batches.

What this script does:
1) Runs eval_run_baselines_v2.py on GPT prediction JSONL files to create baseline eval JSONs.
2) Reads Gemma/Qwen baseline eval paths from existing *_vs_method_batches.json reports.
3) Compares each baseline against method batch eval JSONs (batch_1/2/3) with explicit batch labels.
4) Exports unified JSON and CSV containing:
   - f1
   - EM (exact_exec_match)
   - confidence (pred_confidence)
   - sql ast (sql_ast_similarity)
   - chrf
   - rouge (rouge_l_f1)
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("em", "exact_exec_match", "bool"),
    ("f1", "f1", "num"),
    ("confidence", "pred_confidence", "num"),
    ("sql_ast", "sql_ast_similarity", "num"),
    ("chrf", "chrf", "num"),
    ("rouge", "rouge_l_f1", "num"),
]


@dataclass
class BaselineEntry:
    family: str
    baseline: str
    baseline_eval_path: Path
    source_type: str
    source_path: str


@dataclass
class BatchEntry:
    label: str
    path: Path
    rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent

    ap = argparse.ArgumentParser(description="Unified batch eval for GPT/Gemma/Qwen vs method batches.")
    ap.add_argument(
        "--gpt-preds",
        nargs="+",
        default=[
            str(root / "results" / "gpt_predictions" / "zero_shot" / "predictions.jsonl"),
            str(root / "results" / "gpt_predictions" / "few_shot" / "predictions.jsonl"),
            str(root / "results" / "gpt_predictions" / "cot" / "predictions.jsonl"),
        ],
        help="GPT prediction JSONL files.",
    )
    ap.add_argument(
        "--gemma-vs-json",
        nargs="+",
        default=[
            str(root / "results" / "gemma_method_vs_baselines" / "eval_gemma27b_cot_vs_method_batches.json"),
            str(root / "results" / "gemma_method_vs_baselines" / "eval_gemma27b_few_shot_vs_method_batches.json"),
            str(root / "results" / "gemma_method_vs_baselines" / "eval_gemma27b_zero_shot_vs_method_batches.json"),
        ],
        help="Gemma *_vs_method_batches.json files.",
    )
    ap.add_argument(
        "--qwen-vs-json",
        nargs="+",
        default=[
            str(root / "results" / "qwen_method_vs_baselines" / "eval_qwen3_30b_a3b_cot_vs_method_batches.json"),
            str(root / "results" / "qwen_method_vs_baselines" / "eval_qwen3_30b_a3b_few_shot_vs_method_batches.json"),
            str(root / "results" / "qwen_method_vs_baselines" / "eval_qwen3_30b_a3b_zero_shot_vs_method_batches.json"),
        ],
        help="Qwen *_vs_method_batches.json files.",
    )
    ap.add_argument(
        "--method-batch-evals",
        nargs="+",
        default=[
            str(root / "method" / "batch" / "batch_3" / "batch_3.eval.json"),
            str(root / "method" / "batch" / "batch_2" / "batch_2.eval.json"),
            str(root / "method" / "batch" / "batch_1" / "batch_1.eval.json"),
        ],
        help="Method batch eval JSON files.",
    )
    ap.add_argument(
        "--eval-script",
        default=str(root / "eval_run_baselines_v2.py"),
        help="Path to eval_run_baselines_v2.py (used for GPT predictions).",
    )
    ap.add_argument("--gt-path", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument(
        "--id-source",
        choices=["pred_nonempty", "all"],
        default="pred_nonempty",
        help="Question set per batch: rows with non-empty pred_sql, or all rows.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(root / "results" / "unified_batch_eval"),
        help="Directory for unified outputs.",
    )
    ap.add_argument(
        "--reuse-gpt-evals",
        action="store_true",
        help="Reuse existing GPT eval outputs in output-dir/gpt_eval_outputs instead of re-running eval_run_baselines_v2.py.",
    )
    return ap.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def load_eval(path: Path) -> Dict[str, Any]:
    obj = load_json(path)
    rows = obj.get("per_item")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid eval JSON (missing per_item list): {path}")
    return obj


def to_item_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        iid = str(r.get("item_id") or "").strip()
        if iid:
            out[iid] = r
    return out


def nonempty_pred_sql(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("pred_sql") or "").strip())


def scope_ids_from_rows(rows: Sequence[Dict[str, Any]], id_source: str) -> List[str]:
    if id_source == "pred_nonempty":
        ids = [str(r.get("item_id")) for r in rows if isinstance(r, dict) and nonempty_pred_sql(r)]
    else:
        ids = [str(r.get("item_id")) for r in rows if isinstance(r, dict)]
    return [x for x in ids if x and x != "None"]


def dedupe_keep_order(ids: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for iid in ids:
        if iid in seen:
            continue
        seen.add(iid)
        out.append(iid)
    return out


def mean_optional(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def get_num(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def get_bool(row: Dict[str, Any], key: str) -> int:
    return 1 if bool(row.get(key)) else 0


def baseline_label_from_eval(path: Path) -> str:
    stem = path.stem
    if stem.startswith("eval_"):
        stem = stem[len("eval_") :]
    return stem


def batch_label_from_path(path: Path) -> str:
    parent = path.parent.name.strip()
    if parent:
        return parent
    stem = path.stem
    if stem.endswith(".eval"):
        stem = stem[: -len(".eval")]
    return stem


def run_gpt_eval(
    *,
    eval_script: Path,
    pred_path: Path,
    gt_path: Path,
    db_path: Path,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--pred_path",
        str(pred_path),
        "--gt_path",
        str(gt_path),
        "--db_path",
        str(db_path),
        "--output_json",
        str(output_path),
    ]
    print("[GPT eval]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def collect_baseline_entries_from_vs(vs_paths: Sequence[Path], family: str) -> List[BaselineEntry]:
    entries: List[BaselineEntry] = []
    for vs_path in vs_paths:
        obj = load_json(vs_path)
        baseline_eval = obj.get("baseline_eval")
        if not isinstance(baseline_eval, str) or not baseline_eval.strip():
            raise ValueError(f"Missing baseline_eval in: {vs_path}")
        baseline_path = Path(baseline_eval)
        entries.append(
            BaselineEntry(
                family=family,
                baseline=baseline_label_from_eval(baseline_path),
                baseline_eval_path=baseline_path,
                source_type="vs_json",
                source_path=str(vs_path),
            )
        )
    return entries


def metric_pair(
    *,
    common_ids: Sequence[str],
    method_map: Dict[str, Dict[str, Any]],
    baseline_map: Dict[str, Dict[str, Any]],
    key: str,
    kind: str,
) -> Dict[str, Optional[float]]:
    if kind == "bool":
        m = mean_optional(get_bool(method_map[i], key) for i in common_ids)
        b = mean_optional(get_bool(baseline_map[i], key) for i in common_ids)
    else:
        m = mean_optional(v for v in (get_num(method_map[i], key) for i in common_ids) if v is not None)
        b = mean_optional(v for v in (get_num(baseline_map[i], key) for i in common_ids) if v is not None)

    delta: Optional[float]
    if m is None or b is None:
        delta = None
    else:
        delta = m - b

    return {
        "method": m,
        "baseline": b,
        "delta_method_minus_baseline": delta,
    }


def compare_single_batch(
    *,
    batch_label: str,
    batch_rows: Sequence[Dict[str, Any]],
    baseline_map: Dict[str, Dict[str, Any]],
    id_source: str,
) -> Dict[str, Any]:
    scope_raw = scope_ids_from_rows(batch_rows, id_source)
    scope_ids = dedupe_keep_order(scope_raw)
    method_map = to_item_map(batch_rows)

    common_ids = [i for i in scope_ids if i in method_map and i in baseline_map]
    missing_in_baseline = [i for i in scope_ids if i not in baseline_map]

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for out_name, key, kind in METRIC_SPECS:
        metrics[out_name] = metric_pair(
            common_ids=common_ids,
            method_map=method_map,
            baseline_map=baseline_map,
            key=key,
            kind=kind,
        )

    both_true = method_only_true = baseline_only_true = both_false = 0
    for iid in common_ids:
        m = bool(method_map[iid].get("exact_exec_match"))
        b = bool(baseline_map[iid].get("exact_exec_match"))
        if m and b:
            both_true += 1
        elif m and not b:
            method_only_true += 1
        elif b and not m:
            baseline_only_true += 1
        else:
            both_false += 1

    return {
        "batch": batch_label,
        "question_set_source": id_source,
        "counts": {
            "batch_scope_ids_raw": len(scope_raw),
            "batch_scope_ids": len(scope_ids),
            "common_ids": len(common_ids),
            "missing_in_baseline": len(missing_in_baseline),
            "deduped_scope_ids": len(scope_raw) - len(scope_ids),
        },
        "exact_head_to_head": {
            "both_true": both_true,
            "method_only_true": method_only_true,
            "baseline_only_true": baseline_only_true,
            "both_false": both_false,
        },
        "metrics": metrics,
        "sample_common_ids": common_ids[:10],
    }


def compare_overall(
    *,
    method_batches: Sequence[BatchEntry],
    baseline_map: Dict[str, Dict[str, Any]],
    id_source: str,
) -> Dict[str, Any]:
    scope_raw: List[str] = []
    first_row_by_id: Dict[str, Dict[str, Any]] = {}
    first_batch_by_id: Dict[str, str] = {}

    for batch in method_batches:
        ids = scope_ids_from_rows(batch.rows, id_source)
        row_map = to_item_map(batch.rows)
        scope_raw.extend(ids)
        for iid in ids:
            if iid in first_row_by_id:
                continue
            row = row_map.get(iid)
            if row is None:
                continue
            first_row_by_id[iid] = row
            first_batch_by_id[iid] = batch.label

    scope_ids = dedupe_keep_order(scope_raw)
    common_ids = [i for i in scope_ids if i in baseline_map and i in first_row_by_id]
    missing_in_baseline = [i for i in scope_ids if i not in baseline_map]

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for out_name, key, kind in METRIC_SPECS:
        metrics[out_name] = metric_pair(
            common_ids=common_ids,
            method_map=first_row_by_id,
            baseline_map=baseline_map,
            key=key,
            kind=kind,
        )

    both_true = method_only_true = baseline_only_true = both_false = 0
    for iid in common_ids:
        m = bool(first_row_by_id[iid].get("exact_exec_match"))
        b = bool(baseline_map[iid].get("exact_exec_match"))
        if m and b:
            both_true += 1
        elif m and not b:
            method_only_true += 1
        elif b and not m:
            baseline_only_true += 1
        else:
            both_false += 1

    return {
        "batch": "overall_combined",
        "question_set_source": id_source,
        "counts": {
            "batch_scope_ids_raw": len(scope_raw),
            "batch_scope_ids": len(scope_ids),
            "common_ids": len(common_ids),
            "missing_in_baseline": len(missing_in_baseline),
            "deduped_scope_ids": len(scope_raw) - len(scope_ids),
        },
        "exact_head_to_head": {
            "both_true": both_true,
            "method_only_true": method_only_true,
            "baseline_only_true": baseline_only_true,
            "both_false": both_false,
        },
        "metrics": metrics,
        "sample_common_ids": common_ids[:10],
        "sample_id_sources": {iid: first_batch_by_id.get(iid) for iid in common_ids[:10]},
    }


def build_flat_rows(run_reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for rr in run_reports:
        common_base = {
            "family": rr["family"],
            "baseline": rr["baseline"],
            "baseline_eval": rr["baseline_eval"],
            "source_type": rr["source_type"],
            "source_path": rr["source_path"],
            "id_source": rr["id_source"],
        }

        for result in list(rr["results"]) + [rr["overall"]]:
            out: Dict[str, Any] = dict(common_base)
            out["batch"] = result["batch"]

            counts = result.get("counts", {})
            out["batch_scope_ids"] = counts.get("batch_scope_ids")
            out["batch_scope_ids_raw"] = counts.get("batch_scope_ids_raw")
            out["common_ids"] = counts.get("common_ids")
            out["missing_in_baseline"] = counts.get("missing_in_baseline")
            out["deduped_scope_ids"] = counts.get("deduped_scope_ids")

            h2h = result.get("exact_head_to_head", {})
            out["both_true"] = h2h.get("both_true")
            out["method_only_true"] = h2h.get("method_only_true")
            out["baseline_only_true"] = h2h.get("baseline_only_true")
            out["both_false"] = h2h.get("both_false")

            for metric_name, _, _ in METRIC_SPECS:
                m = result.get("metrics", {}).get(metric_name, {})
                out[f"{metric_name}_method"] = m.get("method")
                out[f"{metric_name}_baseline"] = m.get("baseline")
                out[f"{metric_name}_delta_method_minus_baseline"] = m.get("delta_method_minus_baseline")

            rows.append(out)

    return rows


def save_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "family",
        "baseline",
        "baseline_eval",
        "source_type",
        "source_path",
        "id_source",
        "batch",
        "batch_scope_ids",
        "batch_scope_ids_raw",
        "common_ids",
        "missing_in_baseline",
        "deduped_scope_ids",
        "both_true",
        "method_only_true",
        "baseline_only_true",
        "both_false",
    ]
    for metric_name, _, _ in METRIC_SPECS:
        fieldnames.extend(
            [
                f"{metric_name}_method",
                f"{metric_name}_baseline",
                f"{metric_name}_delta_method_minus_baseline",
            ]
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def require_exists(paths: Sequence[Path], what: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {what}: {missing}")


def main() -> None:
    args = parse_args()

    eval_script = Path(args.eval_script)
    gt_path = Path(args.gt_path)
    db_path = Path(args.db_path)

    gpt_preds = [Path(p) for p in args.gpt_preds]
    gemma_vs = [Path(p) for p in args.gemma_vs_json]
    qwen_vs = [Path(p) for p in args.qwen_vs_json]
    method_batch_paths = [Path(p) for p in args.method_batch_evals]

    require_exists([eval_script, gt_path, db_path], "core inputs")
    require_exists(gpt_preds, "GPT prediction files")
    require_exists(gemma_vs, "Gemma vs JSON files")
    require_exists(qwen_vs, "Qwen vs JSON files")
    require_exists(method_batch_paths, "method batch eval files")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpt_eval_dir = output_dir / "gpt_eval_outputs"
    gpt_eval_dir.mkdir(parents=True, exist_ok=True)

    method_batches: List[BatchEntry] = []
    for p in method_batch_paths:
        obj = load_eval(p)
        rows = obj.get("per_item")
        assert isinstance(rows, list)
        method_batches.append(BatchEntry(label=batch_label_from_path(p), path=p, rows=rows))

    baseline_entries: List[BaselineEntry] = []

    for pred_path in gpt_preds:
        mode = pred_path.parent.name
        eval_out = gpt_eval_dir / f"eval_gpt5nano_{mode}.json"
        if args.reuse_gpt_evals and eval_out.exists():
            print(f"[GPT eval] Reusing existing: {eval_out}")
        else:
            run_gpt_eval(
                eval_script=eval_script,
                pred_path=pred_path,
                gt_path=gt_path,
                db_path=db_path,
                output_path=eval_out,
            )
        baseline_entries.append(
            BaselineEntry(
                family="gpt5nano",
                baseline=mode,
                baseline_eval_path=eval_out,
                source_type="gpt_predictions",
                source_path=str(pred_path),
            )
        )

    baseline_entries.extend(collect_baseline_entries_from_vs(gemma_vs, family="gemma"))
    baseline_entries.extend(collect_baseline_entries_from_vs(qwen_vs, family="qwen"))

    require_exists([b.baseline_eval_path for b in baseline_entries], "baseline eval files")

    run_reports: List[Dict[str, Any]] = []
    for b in baseline_entries:
        print("=" * 100)
        print(f"Comparing family={b.family} baseline={b.baseline}")
        print(f"Baseline eval: {b.baseline_eval_path}")

        baseline_obj = load_eval(b.baseline_eval_path)
        baseline_rows = baseline_obj.get("per_item")
        assert isinstance(baseline_rows, list)
        baseline_map = to_item_map(baseline_rows)

        per_batch_results: List[Dict[str, Any]] = []
        for batch in method_batches:
            res = compare_single_batch(
                batch_label=batch.label,
                batch_rows=batch.rows,
                baseline_map=baseline_map,
                id_source=args.id_source,
            )
            per_batch_results.append(res)

            f1d = (res.get("metrics", {}).get("f1", {}) or {}).get("delta_method_minus_baseline")
            emd = (res.get("metrics", {}).get("em", {}) or {}).get("delta_method_minus_baseline")
            confd = (res.get("metrics", {}).get("confidence", {}) or {}).get("delta_method_minus_baseline")
            print(
                f"  {batch.label}: common_ids={res['counts']['common_ids']} "
                f"| EM_delta={f'{emd:+.4f}' if isinstance(emd, (int, float)) else 'NA'} "
                f"| F1_delta={f'{f1d:+.4f}' if isinstance(f1d, (int, float)) else 'NA'} "
                f"| CONF_delta={f'{confd:+.4f}' if isinstance(confd, (int, float)) else 'NA'}"
            )

        overall = compare_overall(
            method_batches=method_batches,
            baseline_map=baseline_map,
            id_source=args.id_source,
        )

        run_reports.append(
            {
                "family": b.family,
                "baseline": b.baseline,
                "baseline_eval": str(b.baseline_eval_path),
                "source_type": b.source_type,
                "source_path": b.source_path,
                "id_source": args.id_source,
                "method_batches": [
                    {"label": mb.label, "path": str(mb.path)} for mb in method_batches
                ],
                "results": per_batch_results,
                "overall": overall,
            }
        )

    flat_rows = build_flat_rows(run_reports)

    out_json = output_dir / "unified_batch_eval.json"
    out_csv = output_dir / "unified_batch_eval.csv"

    out_json.write_text(
        json.dumps(
            {
                "meta": {
                    "eval_script": str(eval_script),
                    "gt_path": str(gt_path),
                    "db_path": str(db_path),
                    "id_source": args.id_source,
                    "gpt_preds": [str(p) for p in gpt_preds],
                    "gemma_vs_json": [str(p) for p in gemma_vs],
                    "qwen_vs_json": [str(p) for p in qwen_vs],
                    "method_batch_evals": [str(p) for p in method_batch_paths],
                },
                "runs": run_reports,
                "flat_rows": flat_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_csv(flat_rows, out_csv)

    print("=" * 100)
    print(f"Saved unified JSON: {out_json}")
    print(f"Saved unified CSV:  {out_csv}")


if __name__ == "__main__":
    main()
