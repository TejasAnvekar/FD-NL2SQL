#!/usr/bin/env python3
"""Strategy-testing staged orchestrator with cleaner seed growth and fixed holdout tracking.

This entrypoint keeps the main orchestrator untouched and changes only the staged protocol:
- evaluates a fixed holdout slice after each stage
- appends only accepted main-metric rows into the next seed set
- skips fallback-modified SQL when growing the seed set
- uses original model SQL for seed growth by default
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import orchestrate_decompose_retrieve_synthesize as base


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, argparse.Namespace]:
    ap = argparse.ArgumentParser(
        description="Run staged seed growth with fixed holdout evaluation and stricter seed promotion.",
        allow_abbrev=False,
    )
    ap.add_argument(
        "--holdout-question-json",
        default="",
        help="Optional question dataset for holdout evaluation. Defaults to the main --question-json.",
    )
    ap.add_argument(
        "--holdout-gt-json",
        default="",
        help="Optional GT dataset for holdout evaluation. Defaults to the main --gt-json.",
    )
    ap.add_argument("--holdout-start-index", type=int, default=0, help="Start index for the fixed holdout slice.")
    ap.add_argument("--holdout-limit", type=int, default=200, help="Holdout slice size; -1 means all remaining rows.")
    ap.add_argument(
        "--run-holdout-baseline",
        type=int,
        default=1,
        help="1=run holdout eval before stage 1 using the initial seed set.",
    )
    ap.add_argument(
        "--seed-main-metric-only",
        type=int,
        default=1,
        help="1=promote only rows accepted on the main metric, not non-main acceptance.",
    )
    ap.add_argument(
        "--exclude-fallback-from-seed",
        type=int,
        default=1,
        help="1=do not append rows whose final_sql came from fallback post-processing.",
    )
    ap.add_argument(
        "--seed-use-original-sql",
        type=int,
        default=1,
        help="1=append original model SQL instead of fallback-adjusted final_sql.",
    )
    ap.add_argument(
        "--max-seed-additions-per-stage",
        type=int,
        default=0,
        help="Optional cap on promoted rows per stage after filtering; 0 means no cap.",
    )
    ap.add_argument(
        "--summary-json-name",
        default="strategy_holdout_summary.json",
        help="Summary JSON filename written under --batch-root.",
    )
    ap.add_argument(
        "--summary-md-name",
        default="strategy_holdout_summary.md",
        help="Summary Markdown filename written under --batch-root.",
    )

    strategy_args, remaining = ap.parse_known_args(list(argv) if argv is not None else None)

    orig_argv = sys.argv[:]
    try:
        sys.argv = [orig_argv[0]] + remaining
        base_args = base.parse_args()
    finally:
        sys.argv = orig_argv

    return strategy_args, base_args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


TEXT_EQUALITY_PATTERN = re.compile(
    r"(?P<lhs>(?:(?:\"[^\"]+\")|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\"[^\"]+\"|[A-Za-z_][\w$]*))?)\s*=\s*(?P<rhs>'(?:''|[^'])*')",
    flags=re.IGNORECASE,
)


def normalize_sql_terminal(sql: str) -> str:
    s = str(sql or "").strip()
    if not s:
        return ""
    return s.rstrip(";").strip() + ";"


def decode_sql_string_literal(token: str) -> str:
    t = str(token or "").strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        t = t[1:-1]
    return t.replace("''", "'")


def quote_identifier(name: str) -> str:
    return '"' + str(name or "").replace('"', '""') + '"'


def parse_table_name(sql: str) -> str:
    m = re.search(r"\bfrom\b\s+([A-Za-z_][\w$]*)", str(sql or ""), flags=re.IGNORECASE)
    if not m:
        return "clinical_trials"
    return m.group(1)


def abstract_template_signature(sql: str) -> str:
    s = normalize_sql_terminal(sql).lower()
    s = re.sub(r"'(?:''|[^'])*'", "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_text_equalities(sql: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for m in TEXT_EQUALITY_PATTERN.finditer(str(sql or "")):
        lhs = str(m.group("lhs") or "").strip()
        rhs = str(m.group("rhs") or "").strip()
        if not lhs or not rhs:
            continue
        column = lhs.split(".")[-1].strip().strip('"')
        out.append((parse_table_name(sql), column, decode_sql_string_literal(rhs)))
    return out


def distinct_values_for_column(
    *,
    db_path: Path,
    table_name: str,
    column_name: str,
    cache: Dict[Tuple[str, str], set[str]],
) -> set[str]:
    key = (table_name, column_name)
    if key in cache:
        return cache[key]

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            f"SELECT DISTINCT {quote_identifier(column_name)} "
            f"FROM {quote_identifier(table_name)} "
            f"WHERE {quote_identifier(column_name)} IS NOT NULL;"
        )
        cache[key] = {str(row[0]) for row in cur.fetchall() if row and row[0] is not None}
    finally:
        conn.close()
    return cache[key]


def canonical_pairs_for_sql(
    *,
    sql: str,
    db_path: Path,
    distinct_cache: Dict[Tuple[str, str], set[str]],
) -> set[str]:
    pairs: set[str] = set()
    for table_name, column_name, value in extract_text_equalities(sql):
        if value in distinct_values_for_column(
            db_path=db_path,
            table_name=table_name,
            column_name=column_name,
            cache=distinct_cache,
        ):
            pairs.add(f"{table_name}.{column_name}={value}")
    return pairs


def seed_coverage(
    *,
    seed_path: Path,
    db_path: Path,
    distinct_cache: Dict[Tuple[str, str], set[str]],
) -> Tuple[set[str], set[str]]:
    obj = read_json(seed_path)
    if not isinstance(obj, list):
        return set(), set()

    templates: set[str] = set()
    canonical_pairs: set[str] = set()
    for row in obj:
        if not isinstance(row, dict):
            continue
        sql = str(row.get("sql") or row.get("gt_sql") or "").strip()
        if not sql:
            continue
        templates.add(abstract_template_signature(sql))
        canonical_pairs.update(
            canonical_pairs_for_sql(
                sql=sql,
                db_path=db_path,
                distinct_cache=distinct_cache,
            )
        )
    return templates, canonical_pairs


def slice_rows(rows: Sequence[Dict[str, Any]], start: int, limit: int) -> List[Dict[str, Any]]:
    start0 = max(0, int(start))
    if int(limit) == -1:
        return [dict(r) for r in rows[start0:]]
    end0 = start0 + max(0, int(limit))
    return [dict(r) for r in rows[start0:end0]]


def reset_output_overrides(args: argparse.Namespace) -> None:
    args.decompose_output_json = ""
    args.decompose_output_jsonl = ""
    args.synth_output_json = ""
    args.synth_output_jsonl = ""
    args.eval_ready_json = ""
    args.eval_ready_jsonl = ""
    args.eval_output_json = ""
    args.acceptance_output_json = ""


def run_phase(
    *,
    args: argparse.Namespace,
    root: Path,
    seed_json: Path,
    question_json: Path,
    gt_json: Path,
    start_index: int,
    limit: int,
    output_dir: Path,
    run_tag: str,
) -> Dict[str, Any]:
    phase_args = argparse.Namespace(**vars(args))
    phase_args.question = None
    phase_args.batch_mode = 1
    phase_args.question_json = str(question_json)
    phase_args.gt_json = str(gt_json)
    phase_args.seed_json = str(seed_json)
    phase_args.start_index = int(start_index)
    phase_args.limit = int(limit)
    phase_args.output_dir = str(output_dir)
    phase_args.run_tag = str(run_tag)
    reset_output_overrides(phase_args)

    p = base.derive_paths(phase_args, root, create_dirs=not bool(int(args.dry_run)))
    decompose_cmd = base.build_decompose_cmd(phase_args, p)
    base._run(decompose_cmd, dry_run=bool(int(args.dry_run)))

    retrieval_json = Path(p["decompose_json"])
    synth_cmd = base.build_synth_cmd(phase_args, p, retrieval_json)
    base._run(synth_cmd, dry_run=bool(int(args.dry_run)))
    return p


def collect_seed_rows_strategy(
    synth_json_path: Path,
    *,
    base_seed_path: Path,
    db_path: Path,
    main_metric_only: bool,
    exclude_fallback: bool,
    use_original_sql: bool,
    max_seed_additions: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obj = read_json(synth_json_path)
    rows = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return [], {
            "accepted_total": 0,
            "accepted_main_metric": 0,
            "skipped_non_main": 0,
            "skipped_fallback": 0,
            "skipped_no_canonical_pairs": 0,
            "skipped_no_grounding_gain": 0,
            "selected_before_cap": 0,
            "selected_after_cap": 0,
            "new_canonical_pairs_added": 0,
            "new_templates_added": 0,
        }

    distinct_cache: Dict[Tuple[str, str], set[str]] = {}
    existing_templates, existing_canonical_pairs = seed_coverage(
        seed_path=base_seed_path,
        db_path=db_path,
        distinct_cache=distinct_cache,
    )

    candidates: List[Dict[str, Any]] = []
    summary = {
        "accepted_total": 0,
        "accepted_main_metric": 0,
        "skipped_non_main": 0,
        "skipped_fallback": 0,
        "skipped_no_canonical_pairs": 0,
        "skipped_no_grounding_gain": 0,
        "selected_before_cap": 0,
        "selected_after_cap": 0,
        "new_canonical_pairs_added": 0,
        "new_templates_added": 0,
    }

    for r in rows:
        if not isinstance(r, dict) or r.get("error"):
            continue

        acceptance = r.get("acceptance") if isinstance(r.get("acceptance"), dict) else {}
        decision = acceptance.get("decision") if isinstance(acceptance.get("decision"), dict) else {}
        if not bool(decision.get("accepted")):
            continue

        summary["accepted_total"] += 1
        if str(decision.get("reason") or "") == "main_metric":
            summary["accepted_main_metric"] += 1
        elif main_metric_only:
            summary["skipped_non_main"] += 1
            continue

        synth = r.get("synthesized") if isinstance(r.get("synthesized"), dict) else {}
        if exclude_fallback and bool(synth.get("fallback_applied")):
            summary["skipped_fallback"] += 1
            continue

        sql_key = "original_final_sql" if use_original_sql else "final_sql"
        sql = normalize_sql_terminal(str(synth.get(sql_key) or synth.get("final_sql") or "").strip())
        question = str(r.get("input_question") or acceptance.get("question") or "").strip()
        if not question or not sql:
            continue

        template_signature = abstract_template_signature(sql)
        canonical_pairs = canonical_pairs_for_sql(
            sql=sql,
            db_path=db_path,
            distinct_cache=distinct_cache,
        )
        if not canonical_pairs:
            summary["skipped_no_canonical_pairs"] += 1
            continue

        grounding_gain = canonical_pairs - existing_canonical_pairs
        if not grounding_gain:
            summary["skipped_no_grounding_gain"] += 1
            continue

        candidates.append(
            {
                "item_id": r.get("item_id"),
                "question_index": r.get("question_index"),
                "question": question,
                "sql": sql,
                "acceptance_reason": decision.get("reason"),
                "confidence_overall": synth.get("confidence_overall"),
                "fallback_applied": bool(synth.get("fallback_applied")),
                "template_signature": template_signature,
                "canonical_pairs": canonical_pairs,
                "grounding_gain": grounding_gain,
            }
        )

    summary["selected_before_cap"] = len(candidates)

    selected: List[Dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    covered_templates = set(existing_templates)
    covered_pairs = set(existing_canonical_pairs)
    limit = int(max_seed_additions) if int(max_seed_additions) > 0 else len(candidates)

    while len(selected) < limit:
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[float, float, float, float]] = None

        for idx, cand in enumerate(candidates):
            item_id = str(cand.get("item_id") or "")
            if item_id and item_id in seen_item_ids:
                continue

            new_pairs = set(cand["canonical_pairs"]) - covered_pairs
            if not new_pairs:
                continue

            template_is_new = 1.0 if cand["template_signature"] not in covered_templates else 0.0
            confidence = float(cand.get("confidence_overall") or 0.0)
            score = (
                float(len(new_pairs)),
                template_is_new,
                float(len(cand["canonical_pairs"])),
                confidence,
            )
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break

        chosen = candidates.pop(best_idx)
        selected.append(chosen)
        if chosen.get("item_id") is not None:
            seen_item_ids.add(str(chosen.get("item_id")))
        covered_templates.add(str(chosen["template_signature"]))
        covered_pairs.update(set(chosen["canonical_pairs"]))

    summary["selected_after_cap"] = len(selected)
    summary["new_canonical_pairs_added"] = len(covered_pairs - existing_canonical_pairs)
    summary["new_templates_added"] = len(covered_templates - existing_templates)

    out: List[Dict[str, Any]] = []
    for row in selected:
        out.append(
            {
                "item_id": row.get("item_id"),
                "question_index": row.get("question_index"),
                "question": row.get("question"),
                "sql": row.get("sql"),
                "acceptance_reason": row.get("acceptance_reason"),
                "confidence_overall": row.get("confidence_overall"),
                "fallback_applied": row.get("fallback_applied"),
            }
        )

    return out, summary


def extract_eval_summary(eval_json_path: Path) -> Dict[str, Any]:
    obj = read_json(eval_json_path)
    summary = obj.get("summary") if isinstance(obj, dict) and isinstance(obj.get("summary"), dict) else {}
    return {
        "pred_with_sql": int(summary.get("pred_with_sql") or 0),
        "evaluated_items": int(summary.get("evaluated_items") or 0),
        "exec_exact_match": int(summary.get("exec_exact_match") or 0),
        "exec_exact_match_rate": float(summary.get("exec_exact_match_rate") or 0.0),
        "avg_f1": float(summary.get("avg_f1") or 0.0),
        "avg_sql_ast_similarity": float(summary.get("avg_sql_ast_similarity") or 0.0),
    }


def make_holdout_slice_file(
    *,
    source_path: Path,
    start_index: int,
    limit: int,
    out_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    rows = read_json(source_path)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list-form holdout source JSON: {source_path}")

    holdout_rows = slice_rows(rows, start_index, limit)
    if not holdout_rows:
        raise SystemExit(f"Empty holdout slice from {source_path} start={start_index} limit={limit}")

    if not dry_run:
        write_json(out_path, holdout_rows)

    return {
        "path": str(out_path),
        "count": len(holdout_rows),
        "start_index": int(start_index),
        "end_index": int(start_index) + len(holdout_rows) - 1,
    }


def render_summary_markdown(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "# Seed Growth Holdout Summary",
        "",
        "## Strategy",
        "",
        f"- Holdout question JSON: `{summary['holdout']['question_json']}`",
        f"- Holdout GT JSON: `{summary['holdout']['gt_json']}`",
        f"- Holdout range: `{summary['holdout']['start_index']}-{summary['holdout']['end_index']}`",
        f"- Seed main-metric only: `{summary['strategy']['seed_main_metric_only']}`",
        f"- Exclude fallback from seed: `{summary['strategy']['exclude_fallback_from_seed']}`",
        f"- Seed uses original SQL: `{summary['strategy']['seed_use_original_sql']}`",
        f"- Max seed additions per stage: `{summary['strategy']['max_seed_additions_per_stage']}`",
        "",
        "## Holdout Trend",
        "",
        "| Pass | Seed Source | Added | Exec Match | Exec Match Rate | Avg F1 | Avg SQL AST |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    baseline = summary.get("holdout_baseline")
    if isinstance(baseline, dict):
        metrics = baseline.get("eval_summary") if isinstance(baseline.get("eval_summary"), dict) else {}
        lines.append(
            "| "
            f"baseline | {Path(str(baseline.get('seed_json') or '')).name} | 0 | "
            f"{metrics.get('exec_exact_match', 0)}/{metrics.get('evaluated_items', 0)} | "
            f"{float(metrics.get('exec_exact_match_rate') or 0.0):.4f} | "
            f"{float(metrics.get('avg_f1') or 0.0):.4f} | "
            f"{float(metrics.get('avg_sql_ast_similarity') or 0.0):.4f} |"
        )

    for stage in summary.get("stages", []):
        metrics = stage.get("holdout_eval_summary") if isinstance(stage.get("holdout_eval_summary"), dict) else {}
        lines.append(
            "| "
            f"{stage.get('stage')} | {Path(str(stage.get('seed_in') or '')).name} | {stage.get('seed_rows_added', 0)} | "
            f"{metrics.get('exec_exact_match', 0)}/{metrics.get('evaluated_items', 0)} | "
            f"{float(metrics.get('exec_exact_match_rate') or 0.0):.4f} | "
            f"{float(metrics.get('avg_f1') or 0.0):.4f} | "
            f"{float(metrics.get('avg_sql_ast_similarity') or 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Seed Filters",
            "",
            "| Stage | Accepted | Main Metric | Skip Non-Main | Skip Fallback | Skip No Canonical | Skip No Gain | Selected | New Canonical Pairs | New Templates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for stage in summary.get("stages", []):
        filt = stage.get("seed_filter_summary") if isinstance(stage.get("seed_filter_summary"), dict) else {}
        lines.append(
            "| "
            f"{stage.get('stage')} | "
            f"{int(filt.get('accepted_total') or 0)} | "
            f"{int(filt.get('accepted_main_metric') or 0)} | "
            f"{int(filt.get('skipped_non_main') or 0)} | "
            f"{int(filt.get('skipped_fallback') or 0)} | "
            f"{int(filt.get('skipped_no_canonical_pairs') or 0)} | "
            f"{int(filt.get('skipped_no_grounding_gain') or 0)} | "
            f"{int(filt.get('selected_after_cap') or 0)} | "
            f"{int(filt.get('new_canonical_pairs_added') or 0)} | "
            f"{int(filt.get('new_templates_added') or 0)} |"
        )

    return lines


def main() -> None:
    strategy_args, args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if int(args.batch_mode) != 1:
        raise SystemExit("This strategy orchestrator requires --batch-mode=1")
    if str(args.candidate_json).strip() or str(args.candidate_sqlite).strip():
        raise SystemExit("This strategy orchestrator requires seed-based retrieval; do not set --candidate-json/--candidate-sqlite")
    if int(args.skip_decompose) == 1:
        raise SystemExit("This strategy orchestrator does not support --skip-decompose=1")

    batch_root = Path(args.batch_root)
    seeds_dir = batch_root / "seeds"
    holdout_dir = batch_root / "holdout"
    if not bool(int(args.dry_run)):
        batch_root.mkdir(parents=True, exist_ok=True)
        seeds_dir.mkdir(parents=True, exist_ok=True)
        holdout_dir.mkdir(parents=True, exist_ok=True)

    seed_src = Path(args.seed_json)
    seed_working0 = seeds_dir / str(args.seed_copy_name)
    print("Strategy staged mode root:", batch_root)
    print("Initial seed source:", seed_src)
    print("Initial working seed:", seed_working0)
    if not bool(int(args.dry_run)):
        seed_working0.write_text(seed_src.read_text(encoding="utf-8"), encoding="utf-8")

    holdout_question_json = Path(strategy_args.holdout_question_json or args.question_json)
    holdout_gt_json = Path(strategy_args.holdout_gt_json or args.gt_json)
    holdout_gt_slice_path = holdout_dir / "holdout_gt_slice.json"
    holdout_slice = make_holdout_slice_file(
        source_path=holdout_gt_json,
        start_index=int(strategy_args.holdout_start_index),
        limit=int(strategy_args.holdout_limit),
        out_path=holdout_gt_slice_path,
        dry_run=bool(int(args.dry_run)),
    )
    print(
        "Holdout slice:",
        f"{holdout_slice['start_index']}-{holdout_slice['end_index']}",
        f"count={holdout_slice['count']}",
    )

    summary: Dict[str, Any] = {
        "mode": "seed_growth_holdout_strategy",
        "batch_root": str(batch_root),
        "initial_seed": str(seed_src),
        "strategy": {
            "seed_main_metric_only": bool(int(strategy_args.seed_main_metric_only)),
            "exclude_fallback_from_seed": bool(int(strategy_args.exclude_fallback_from_seed)),
            "seed_use_original_sql": bool(int(strategy_args.seed_use_original_sql)),
            "max_seed_additions_per_stage": int(strategy_args.max_seed_additions_per_stage),
            "run_holdout_baseline": bool(int(strategy_args.run_holdout_baseline)),
        },
        "holdout": {
            "question_json": str(holdout_question_json),
            "gt_json": str(holdout_gt_json),
            "gt_slice_path": str(holdout_gt_slice_path),
            "start_index": int(holdout_slice["start_index"]),
            "end_index": int(holdout_slice["end_index"]),
            "count": int(holdout_slice["count"]),
        },
        "holdout_baseline": None,
        "stages": [],
        "final_seed": None,
    }

    current_seed = seed_working0

    if bool(int(strategy_args.run_holdout_baseline)):
        print("\n=== Holdout baseline | initial seed ===")
        holdout_p = run_phase(
            args=args,
            root=root,
            seed_json=current_seed,
            question_json=holdout_question_json,
            gt_json=holdout_gt_slice_path,
            start_index=int(strategy_args.holdout_start_index),
            limit=int(strategy_args.holdout_limit),
            output_dir=holdout_dir / "baseline",
            run_tag="holdout_baseline",
        )
        baseline_metrics = {}
        if not bool(int(args.dry_run)):
            baseline_metrics = extract_eval_summary(Path(holdout_p["eval_output_json"]))
        summary["holdout_baseline"] = {
            "seed_json": str(current_seed),
            "output_dir": str(holdout_p["out_dir"]),
            "eval_output_json": str(holdout_p["eval_output_json"]),
            "eval_summary": baseline_metrics,
        }

    stage_size = max(1, int(args.stage_size))
    stage_count = max(1, int(args.stage_count))
    base_start = int(args.start_index)

    for stage_idx in range(1, stage_count + 1):
        stage_start = base_start + (stage_idx - 1) * stage_size
        stage_limit = stage_size
        stage_end = stage_start + stage_limit - 1
        stage_seed_in = current_seed

        print(
            f"\n=== Stage {stage_idx}/{stage_count} | questions {stage_start}-{stage_end} | seed={current_seed.name} ==="
        )
        train_p = run_phase(
            args=args,
            root=root,
            seed_json=stage_seed_in,
            question_json=Path(args.question_json),
            gt_json=Path(args.gt_json),
            start_index=stage_start,
            limit=stage_limit,
            output_dir=batch_root / f"batch_{stage_idx}",
            run_tag=f"batch_{stage_idx}",
        )

        seed_rows_added = 0
        seed_rows_skipped_existing = 0
        seed_filter_summary: Dict[str, Any] = {
            "accepted_total": 0,
            "accepted_main_metric": 0,
            "skipped_non_main": 0,
            "skipped_fallback": 0,
            "selected_before_cap": 0,
            "selected_after_cap": 0,
        }
        next_seed = seeds_dir / f"seed_after_batch_{stage_idx}.json"

        if bool(int(args.append_accepted_to_seed)):
            if bool(int(args.dry_run)):
                print(f"[dry-run] Would filter accepted rows from {train_p['synth_json']} into {next_seed}")
                current_seed = next_seed
            else:
                accepted_rows, seed_filter_summary = collect_seed_rows_strategy(
                    Path(train_p["synth_json"]),
                    base_seed_path=stage_seed_in,
                    db_path=Path(args.db_path),
                    main_metric_only=bool(int(strategy_args.seed_main_metric_only)),
                    exclude_fallback=bool(int(strategy_args.exclude_fallback_from_seed)),
                    use_original_sql=bool(int(strategy_args.seed_use_original_sql)),
                    max_seed_additions=int(strategy_args.max_seed_additions_per_stage),
                )
                seed_rows_added, seed_rows_skipped_existing = base._write_seed_with_appends(
                    base_seed_path=current_seed,
                    out_seed_path=next_seed,
                    accepted_rows=accepted_rows,
                    stage_idx=stage_idx,
                    dry_run=False,
                )
                current_seed = next_seed

        print(f"=== Holdout after stage {stage_idx} | seed={current_seed.name} ===")
        holdout_p = run_phase(
            args=args,
            root=root,
            seed_json=current_seed,
            question_json=holdout_question_json,
            gt_json=holdout_gt_slice_path,
            start_index=int(strategy_args.holdout_start_index),
            limit=int(strategy_args.holdout_limit),
            output_dir=holdout_dir / f"stage_{stage_idx}",
            run_tag=f"holdout_stage_{stage_idx}",
        )
        holdout_metrics = {}
        if not bool(int(args.dry_run)):
            holdout_metrics = extract_eval_summary(Path(holdout_p["eval_output_json"]))

        summary["stages"].append(
            {
                "stage": stage_idx,
                "start_index": stage_start,
                "end_index": stage_end,
                "train_output_dir": str(train_p["out_dir"]),
                "train_synth_json": str(train_p["synth_json"]),
                "train_eval_output_json": str(train_p["eval_output_json"]),
                "seed_in": str(stage_seed_in),
                "seed_out": str(next_seed) if bool(int(args.append_accepted_to_seed)) else str(current_seed),
                "seed_filter_summary": seed_filter_summary,
                "seed_rows_added": seed_rows_added,
                "seed_rows_skipped_existing": seed_rows_skipped_existing,
                "holdout_output_dir": str(holdout_p["out_dir"]),
                "holdout_eval_output_json": str(holdout_p["eval_output_json"]),
                "holdout_eval_summary": holdout_metrics,
            }
        )

    summary["final_seed"] = str(current_seed)

    summary_json_path = batch_root / str(strategy_args.summary_json_name)
    summary_md_path = batch_root / str(strategy_args.summary_md_name)
    if not bool(int(args.dry_run)):
        write_json(summary_json_path, summary)
        write_markdown(summary_md_path, render_summary_markdown(summary))

    print("\nStrategy pipeline complete")
    print("Summary JSON:", summary_json_path)
    print("Summary Markdown:", summary_md_path)


if __name__ == "__main__":
    main()
