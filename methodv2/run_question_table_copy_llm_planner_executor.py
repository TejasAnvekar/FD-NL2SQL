#!/usr/bin/env python3
"""Two-stage planner+executor LLM pipeline for table-copy derivation questions."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_hidden_column_sql_eval import make_run_dir, sanitize_name  # noqa: E402
from run_question_table_copy_llm_naive import (  # noqa: E402
    assign_ground_truth_row_ids,
    canonical_jsonable,
    choose_hidden_columns,
    completion_meta,
    drop_columns,
    exact_match,
    extract_payload_values,
    fetch_full_table,
    load_sample_questions,
    parse_chat_completion_text,
    parse_jsonish,
    print_progress,
    run_one_call_with_retries,
    table_to_csv_text,
    value_match,
)
from utils import setup_logger, write_json  # noqa: E402


DEFAULT_MANIFEST = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv"
DEFAULT_ANNOTATED_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv"
DEFAULT_DB_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"
DEFAULT_PROMPT_DIR = THIS_DIR / "prompts"


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return (sum(values) / len(values)) if values else 0.0


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a planner+executor table-copy LLM pipeline. The selector executor "
            "first finds candidate rows, the planner creates a derivation plan, and "
            "the final executor derives the answer for each selected row."
        )
    )
    ap.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    ap.add_argument("--annotated_csv", default=DEFAULT_ANNOTATED_CSV)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="question_table_copy_llm_planner_executor")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--csv_row_number", type=int, default=0)
    ap.add_argument(
        "--resume",
        type=int,
        default=1,
        help=(
            "Resume from completed per-question artifacts already present in the "
            "run directory. Questions with successful completed artifacts are skipped."
        ),
    )

    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--planner_model_name", default="")
    ap.add_argument("--executor_api_base", default="")
    ap.add_argument("--executor_api_key", default="")
    ap.add_argument("--executor_model_name", default="")
    ap.add_argument("--planner_api_base", default="")
    ap.add_argument("--planner_api_key", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--selector_max_tokens", type=int, default=768)
    ap.add_argument("--planner_max_tokens", type=int, default=1024)
    ap.add_argument("--executor_max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument("--max_in_flight", type=int, default=1)
    ap.add_argument("--planner_preview_rows", type=int, default=8)
    ap.add_argument("--planner_input_cost_per_million_tokens", type=float, default=-1.0)
    ap.add_argument("--planner_output_cost_per_million_tokens", type=float, default=-1.0)
    ap.add_argument("--executor_input_cost_per_million_tokens", type=float, default=-1.0)
    ap.add_argument("--executor_output_cost_per_million_tokens", type=float, default=-1.0)
    ap.add_argument("--cost_currency", default="USD")

    ap.add_argument(
        "--answer_leak_columns",
        nargs="*",
        default=[],
        help=(
            "Optional extra table columns to always hide when present. This is "
            "in addition to any per-row annotated ground-truth/final columns "
            "that are distinct from the source column."
        ),
    )
    ap.add_argument(
        "--selector_prompt_file",
        default=str(DEFAULT_PROMPT_DIR / "planner_executor_selector_csv.txt"),
    )
    ap.add_argument(
        "--planner_prompt_file",
        default=str(DEFAULT_PROMPT_DIR / "planner_executor_planner.txt"),
    )
    ap.add_argument(
        "--executor_prompt_file",
        default=str(DEFAULT_PROMPT_DIR / "planner_executor_executor_csv.txt"),
    )
    return ap.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_template(template_text: str, replacements: Dict[str, Any]) -> str:
    out = template_text
    for key, value in replacements.items():
        out = out.replace("{{" + key + "}}", "" if value is None else str(value))
    return out.strip()


def parse_json_block(raw_text: str) -> Any:
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def parse_selector_output(raw_text: str) -> Dict[str, Any]:
    parsed = parse_json_block(raw_text)
    selected_row_ids: List[int] = []
    selection_reason = ""
    needs_derivation = True
    if isinstance(parsed, dict):
        raw_ids = parsed.get("selected_row_ids") or parsed.get("row_ids") or parsed.get("selected_rows") or []
        if isinstance(raw_ids, list):
            for item in raw_ids:
                try:
                    selected_row_ids.append(int(item))
                except Exception:
                    continue
        selection_reason = str(parsed.get("selection_reason") or parsed.get("reason") or "").strip()
        if parsed.get("needs_derivation") is not None:
            needs_derivation = bool(parsed.get("needs_derivation"))
    return {
        "selected_row_ids": sorted(dict.fromkeys(selected_row_ids)),
        "selection_reason": selection_reason,
        "needs_derivation": needs_derivation,
        "raw_parsed_json": parsed,
    }


def build_fallback_plan(*, question: str, visible_cols: Sequence[str]) -> Dict[str, Any]:
    return {
        "task_type": "derive_answer_for_selected_rows",
        "inferred_source_column": "",
        "derived_field_description": question,
        "answer_type": "string_or_boolean_or_list",
        "relevant_columns": list(visible_cols),
        "row_filter_restatement": question,
        "derivation_rules": [
            "Infer which visible column contains the source evidence for the question.",
            "Use likely source values and nearby visible context to derive the answer.",
            "Do not assume the answer already appears verbatim in the table.",
        ],
        "normalization_rules": [
            "Use canonical labels when obvious from the question wording.",
        ],
        "output_constraints": [
            "Return one prediction per selected row.",
            "Use the row __rowid__ as table_row_id.",
        ],
        "notes": "Fallback plan because the planner output was unavailable or invalid.",
    }


def parse_planner_output(raw_text: str, *, question: str, visible_cols: Sequence[str]) -> Dict[str, Any]:
    parsed = parse_json_block(raw_text)
    if not isinstance(parsed, dict):
        return build_fallback_plan(question=question, visible_cols=visible_cols)
    inferred_source_column = str(parsed.get("inferred_source_column") or "").strip()
    if inferred_source_column not in visible_cols:
        inferred_source_column = ""
    relevant_columns = parsed.get("relevant_columns")
    if not isinstance(relevant_columns, list):
        relevant_columns = []
    cleaned_relevant: List[str] = []
    seen = set()
    for col in ["__rowid__", "NCT", "PubMed ID", "Trial name"]:
        if col in visible_cols and col not in seen:
            seen.add(col)
            cleaned_relevant.append(col)
    if inferred_source_column and inferred_source_column not in seen:
        seen.add(inferred_source_column)
        cleaned_relevant.append(inferred_source_column)
    for col in relevant_columns:
        if isinstance(col, str) and col in visible_cols and col not in seen:
            seen.add(col)
            cleaned_relevant.append(col)
    if not cleaned_relevant:
        cleaned_relevant = list(visible_cols)
    return {
        "task_type": str(parsed.get("task_type") or "derive_answer_for_selected_rows"),
        "inferred_source_column": inferred_source_column,
        "derived_field_description": str(parsed.get("derived_field_description") or question),
        "answer_type": str(parsed.get("answer_type") or "string_or_boolean_or_list"),
        "relevant_columns": cleaned_relevant,
        "row_filter_restatement": str(parsed.get("row_filter_restatement") or question),
        "derivation_rules": [str(x) for x in (parsed.get("derivation_rules") or []) if str(x).strip()],
        "normalization_rules": [str(x) for x in (parsed.get("normalization_rules") or []) if str(x).strip()],
        "output_constraints": [str(x) for x in (parsed.get("output_constraints") or []) if str(x).strip()],
        "notes": str(parsed.get("notes") or ""),
    }


def parse_final_predictions(raw_text: str) -> List[Dict[str, Any]]:
    parsed = parse_json_block(raw_text)
    if isinstance(parsed, dict):
        preds = parsed.get("predictions")
        if not isinstance(preds, list):
            preds = [parsed]
    elif isinstance(parsed, list):
        preds = parsed
    else:
        preds = []
    out: List[Dict[str, Any]] = []
    for item in preds:
        if not isinstance(item, dict):
            continue
        row_id = item.get("table_row_id")
        if row_id is None:
            for key in ("row_id", "__rowid__", "row_index"):
                if key in item:
                    row_id = item[key]
                    break
        try:
            row_id_int = int(row_id)
        except Exception:
            continue
        answer = item.get("answer")
        if answer is None:
            for key in ("predicted_value", "prediction", "value"):
                if key in item:
                    answer = item[key]
                    break
        out.append({"table_row_id": row_id_int, "answer": canonical_jsonable(answer)})
    return out


def project_rows(rows: Sequence[Dict[str, Any]], keep_columns: Sequence[str]) -> List[Dict[str, Any]]:
    keep = [col for col in keep_columns if col]
    return [{col: row.get(col) for col in keep if col in row} for row in rows]


def filter_rows_by_ids(rows: Sequence[Dict[str, Any]], row_ids: Sequence[int]) -> List[Dict[str, Any]]:
    wanted = {int(x) for x in row_ids}
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_id = row.get("__rowid__")
        try:
            row_id_int = int(row_id)
        except Exception:
            continue
        if row_id_int in wanted:
            out.append(dict(row))
    return out


def candidate_preview_json(
    rows: Sequence[Dict[str, Any]],
    *,
    visible_cols: Sequence[str],
    max_rows: int,
    max_columns: int = 8,
) -> str:
    id_cols = ["__rowid__", "NCT", "PubMed ID", "Trial name"]
    extra_cols = [col for col in visible_cols if col not in id_cols]
    cols = [col for col in (id_cols + extra_cols[: max(0, int(max_columns) - len(id_cols))]) if col in visible_cols]
    projected = project_rows(rows[: max_rows], cols)
    return json.dumps(projected, ensure_ascii=False, sort_keys=False)


def resolve_stage_endpoint(args: argparse.Namespace, stage: str) -> Dict[str, str]:
    if stage == "planner":
        return {
            "api_base": args.planner_api_base or args.api_base,
            "api_key": args.planner_api_key or args.api_key,
            "model_name": args.planner_model_name or args.executor_model_name or args.model_name,
        }
    return {
        "api_base": args.executor_api_base or args.api_base,
        "api_key": args.executor_api_key or args.api_key,
        "model_name": args.executor_model_name or args.model_name,
    }


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def extract_usage_metrics(model_meta: Dict[str, Any]) -> Dict[str, Optional[int]]:
    prompt_tokens = safe_int(model_meta.get("prompt_tokens"))
    completion_tokens = safe_int(model_meta.get("completion_tokens"))
    total_tokens = safe_int(model_meta.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_available": int(
            prompt_tokens is not None or completion_tokens is not None or total_tokens is not None
        ),
    }


def stage_pricing(args: argparse.Namespace, stage: str) -> Dict[str, float]:
    if stage == "planner":
        return {
            "input_rate": float(args.planner_input_cost_per_million_tokens),
            "output_rate": float(args.planner_output_cost_per_million_tokens),
        }
    return {
        "input_rate": float(args.executor_input_cost_per_million_tokens),
        "output_rate": float(args.executor_output_cost_per_million_tokens),
    }


def estimate_token_cost(
    *,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    input_rate: float,
    output_rate: float,
) -> Dict[str, Optional[float]]:
    input_cost = None
    output_cost = None
    if prompt_tokens is not None and input_rate >= 0:
        input_cost = float(prompt_tokens) * float(input_rate) / 1_000_000.0
    if completion_tokens is not None and output_rate >= 0:
        output_cost = float(completion_tokens) * float(output_rate) / 1_000_000.0
    total_cost = None
    if input_cost is not None or output_cost is not None:
        total_cost = float((input_cost or 0.0) + (output_cost or 0.0))
    return {
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": total_cost,
    }


def combine_usage_metrics(stage_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    prompt_chars = 0
    stages_with_usage = 0
    stage_count = 0
    for metrics in stage_metrics:
        if not metrics:
            continue
        stage_count += 1
        prompt_chars += int(metrics.get("prompt_chars") or 0)
        if metrics.get("usage_available"):
            stages_with_usage += 1
        if metrics.get("prompt_tokens") is not None:
            prompt_tokens += int(metrics["prompt_tokens"])
        if metrics.get("completion_tokens") is not None:
            completion_tokens += int(metrics["completion_tokens"])
        if metrics.get("total_tokens") is not None:
            total_tokens += int(metrics["total_tokens"])
    return {
        "prompt_chars": prompt_chars,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens if total_tokens else prompt_tokens + completion_tokens,
        "stages_with_usage": stages_with_usage,
        "stage_count": stage_count,
        "usage_coverage": (float(stages_with_usage) / float(stage_count)) if stage_count else 0.0,
    }


def combine_cost_metrics(stage_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    input_cost = 0.0
    output_cost = 0.0
    total_cost = 0.0
    saw_any = False
    for metrics in stage_metrics:
        if not metrics:
            continue
        if metrics.get("estimated_input_cost") is not None:
            input_cost += float(metrics["estimated_input_cost"])
            saw_any = True
        if metrics.get("estimated_output_cost") is not None:
            output_cost += float(metrics["estimated_output_cost"])
            saw_any = True
        if metrics.get("estimated_total_cost") is not None:
            total_cost += float(metrics["estimated_total_cost"])
    if not saw_any:
        return {
            "estimated_input_cost": None,
            "estimated_output_cost": None,
            "estimated_total_cost": None,
        }
    return {
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": total_cost,
    }


def sum_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        total += float(value)
    return total


def sum_numeric_or_none(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        vals.append(float(value))
    return float(sum(vals)) if vals else None


def avg_numeric_or_none(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        vals.append(float(value))
    return float(sum(vals) / len(vals)) if vals else None


def question_run_dir(run_dir: Path, item: Dict[str, Any]) -> Path:
    question_slug = sanitize_name(f"{item['item_id']}__{item['question']}")[:180]
    return run_dir / "questions" / question_slug


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_csv_rows(path))


def count_csv_columns(path: Path) -> int:
    if not path.exists():
        return 0
    rows = read_csv_rows(path)
    if rows:
        return len(rows[0].keys())
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return 0
    return len(header)


def stage_response_row(
    *,
    stage: str,
    item: Dict[str, Any],
    question: str,
    raw_text: str,
    parsed_output: Any,
    error: str,
    stage_metrics: Dict[str, Any],
    model_meta: Dict[str, Any],
    cost_currency: str,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "item_id": item["item_id"],
        "csv_row_number": item["csv_row_number"],
        "question": question,
        "llm_raw_output": raw_text,
        "parsed_output_json": json.dumps(parsed_output, ensure_ascii=False),
        "error": error,
        "prompt_char_count": stage_metrics.get("prompt_chars"),
        "prompt_tokens": stage_metrics.get("prompt_tokens"),
        "completion_tokens": stage_metrics.get("completion_tokens"),
        "total_tokens": stage_metrics.get("total_tokens"),
        "usage_available": stage_metrics.get("usage_available"),
        "estimated_input_cost": stage_metrics.get("estimated_input_cost"),
        "estimated_output_cost": stage_metrics.get("estimated_output_cost"),
        "estimated_total_cost": stage_metrics.get("estimated_total_cost"),
        "cost_currency": cost_currency,
        "model_meta_json": json.dumps(model_meta, ensure_ascii=False),
    }


def question_row_has_errors(question_row: Dict[str, Any]) -> bool:
    return bool(
        question_row.get("selector_error")
        or question_row.get("planner_error")
        or question_row.get("executor_error")
    )


def load_resumed_question_result(
    *,
    item: Dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    question_dir = question_run_dir(run_dir, item)
    metadata_path = question_dir / "metadata.json"
    llm_predictions_path = question_dir / "llm_predictions.csv"
    table_copy_path = question_dir / "table_copy.csv"
    candidate_table_path = question_dir / "candidate_table.csv"
    candidate_focused_path = question_dir / "candidate_table_focused.csv"
    selector_prompt_path = question_dir / "selector_prompt.txt"
    planner_prompt_path = question_dir / "planner_prompt.txt"
    executor_prompt_path = question_dir / "executor_prompt.txt"
    selector_response_path = question_dir / "selector_response.json"
    planner_response_path = question_dir / "planner_response.json"
    executor_response_path = question_dir / "executor_response.json"
    ground_truth_with_rowids_path = question_dir / "ground_truth_with_rowids.csv"
    ground_truth_table_path = question_dir / "ground_truth_table.csv"

    required = [
        metadata_path,
        llm_predictions_path,
        table_copy_path,
        candidate_table_path,
        candidate_focused_path,
        selector_prompt_path,
        planner_prompt_path,
        executor_prompt_path,
        selector_response_path,
        planner_response_path,
        ground_truth_with_rowids_path,
        ground_truth_table_path,
    ]
    if any(not path.exists() for path in required):
        return None

    try:
        metadata = read_json(metadata_path)
        llm_rows = read_csv_rows(llm_predictions_path)
        selector_response = read_json(selector_response_path)
        planner_response = read_json(planner_response_path)
        executor_response = read_json(executor_response_path) if executor_response_path.exists() else {}
    except Exception:
        return None

    selector_error = str(metadata.get("selector_error") or "")
    planner_error = str(metadata.get("planner_error") or "")
    executor_error = str(metadata.get("executor_error") or "")
    if selector_error or planner_error or executor_error:
        return None

    question = str(metadata.get("question") or item["question"])
    source_column = str(metadata.get("column_used") or item["column_used"])
    expected_keys = list(metadata.get("expected_keys") or item["expected_keys"])
    hidden_columns = list(metadata.get("hidden_columns") or [])

    selector_stage_metrics = dict(metadata.get("selector_stage_usage") or {})
    planner_stage_metrics = dict(metadata.get("planner_stage_usage") or {})
    executor_stage_metrics = dict(metadata.get("executor_stage_usage") or {})
    executor_side_usage = dict(metadata.get("executor_side_usage") or {})
    planner_side_usage = dict(metadata.get("planner_side_usage") or {})
    all_stage_usage = dict(metadata.get("all_stage_usage") or {})
    executor_side_cost = dict(metadata.get("executor_side_estimated_cost") or {})
    planner_side_cost = dict(metadata.get("planner_side_estimated_cost") or {})
    all_stage_cost = dict(metadata.get("all_stage_estimated_cost") or {})
    executor_endpoint = dict(metadata.get("executor_endpoint") or resolve_stage_endpoint(args, "executor_final"))
    planner_endpoint = dict(metadata.get("planner_endpoint") or resolve_stage_endpoint(args, "planner"))

    gt_rows = [row for row in llm_rows if (row.get("row_type") or "") != "extra_prediction"]
    gt_row_ids = {
        int(row["table_row_id"])
        for row in gt_rows
        if str(row.get("table_row_id", "")).strip()
    }
    pred_row_ids = {
        int(row["table_row_id"])
        for row in llm_rows
        if str(row.get("table_row_id", "")).strip() and str(row.get("predicted_answer_json", "")).strip()
    }
    matched_row_ids = gt_row_ids & pred_row_ids
    matched_exact = sum(
        1
        for row in gt_rows
        if str(row.get("table_row_id", "")).strip() and float(row.get("exact_match") or 0.0) >= 1.0 - 1e-12
    )
    row_level_count = sum(1 for row in gt_rows if str(row.get("table_row_id", "")).strip())
    gt_count = len(gt_row_ids)
    pred_count = len(pred_row_ids)
    exact_match_rate = (matched_exact / row_level_count) if row_level_count else 0.0
    row_recall = (len(matched_row_ids) / gt_count) if gt_count else (1.0 if pred_count == 0 else 0.0)
    row_precision = (len(matched_row_ids) / pred_count) if pred_count else (1.0 if gt_count == 0 else 0.0)
    row_f1 = (2.0 * row_precision * row_recall / (row_precision + row_recall)) if (row_precision + row_recall) > 0 else 0.0
    all_rows_exact_match = 1.0 if gt_count == pred_count and matched_exact == row_level_count and gt_count == len(matched_row_ids) else 0.0

    selector_prompt = selector_prompt_path.read_text(encoding="utf-8")
    planner_prompt = planner_prompt_path.read_text(encoding="utf-8")
    executor_prompt = executor_prompt_path.read_text(encoding="utf-8")

    request_rows = [
        {
            "stage": "selector",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": executor_endpoint.get("api_base", ""),
            "model_name": executor_endpoint.get("model_name", ""),
            "prompt": selector_prompt,
            "prompt_char_count": selector_stage_metrics.get("prompt_chars", len(selector_prompt)),
        },
        {
            "stage": "planner",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": planner_endpoint.get("api_base", ""),
            "model_name": planner_endpoint.get("model_name", ""),
            "prompt": planner_prompt,
            "prompt_char_count": planner_stage_metrics.get("prompt_chars", len(planner_prompt)),
        },
        {
            "stage": "executor_final",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": executor_endpoint.get("api_base", ""),
            "model_name": executor_endpoint.get("model_name", ""),
            "prompt": executor_prompt,
            "prompt_char_count": executor_stage_metrics.get("prompt_chars", len(executor_prompt)),
        },
    ]

    responses_rows = [
        stage_response_row(
            stage="selector",
            item=item,
            question=question,
            raw_text=str(selector_response.get("raw_text") or ""),
            parsed_output=selector_response.get("parsed_output") or metadata.get("selector_output") or {},
            error=str(selector_response.get("error") or ""),
            stage_metrics=selector_stage_metrics,
            model_meta=dict(selector_response.get("model_meta") or metadata.get("selector_model_meta") or {}),
            cost_currency=str(metadata.get("cost_currency") or args.cost_currency),
        ),
        stage_response_row(
            stage="planner",
            item=item,
            question=question,
            raw_text=str(planner_response.get("raw_text") or ""),
            parsed_output=planner_response.get("parsed_plan") or metadata.get("planner_plan") or {},
            error=str(planner_response.get("error") or ""),
            stage_metrics=planner_stage_metrics,
            model_meta=dict(planner_response.get("model_meta") or metadata.get("planner_model_meta") or {}),
            cost_currency=str(metadata.get("cost_currency") or args.cost_currency),
        ),
        stage_response_row(
            stage="executor_final",
            item=item,
            question=question,
            raw_text=str(executor_response.get("raw_text") or ""),
            parsed_output=executor_response.get("parsed_predictions") or metadata.get("parsed_predictions") or [],
            error=str(executor_response.get("error") or metadata.get("executor_error") or ""),
            stage_metrics=executor_stage_metrics,
            model_meta=dict(executor_response.get("model_meta") or metadata.get("executor_model_meta") or {}),
            cost_currency=str(metadata.get("cost_currency") or args.cost_currency),
        ),
    ]

    question_row = {
        "item_id": item["item_id"],
        "csv_row_number": item["csv_row_number"],
        "question": question,
        "column_used": source_column,
        "planner_inferred_source_column": str(metadata.get("planner_inferred_source_column") or ""),
        "planner_inferred_source_matches_metadata": int(metadata.get("planner_inferred_source_matches_metadata") or 0),
        "selector_prompt_chars": selector_stage_metrics.get("prompt_chars"),
        "planner_prompt_chars": planner_stage_metrics.get("prompt_chars"),
        "executor_prompt_chars": executor_stage_metrics.get("prompt_chars"),
        "executor_side_prompt_chars": executor_side_usage.get("prompt_chars"),
        "total_prompt_chars": all_stage_usage.get("prompt_chars"),
        "selector_prompt_tokens": selector_stage_metrics.get("prompt_tokens"),
        "selector_completion_tokens": selector_stage_metrics.get("completion_tokens"),
        "selector_total_tokens": selector_stage_metrics.get("total_tokens"),
        "planner_prompt_tokens": planner_stage_metrics.get("prompt_tokens"),
        "planner_completion_tokens": planner_stage_metrics.get("completion_tokens"),
        "planner_total_tokens": planner_stage_metrics.get("total_tokens"),
        "executor_prompt_tokens": executor_stage_metrics.get("prompt_tokens"),
        "executor_completion_tokens": executor_stage_metrics.get("completion_tokens"),
        "executor_total_tokens": executor_stage_metrics.get("total_tokens"),
        "executor_side_prompt_tokens": executor_side_usage.get("prompt_tokens"),
        "executor_side_completion_tokens": executor_side_usage.get("completion_tokens"),
        "executor_side_total_tokens": executor_side_usage.get("total_tokens"),
        "planner_side_prompt_tokens": planner_side_usage.get("prompt_tokens"),
        "planner_side_completion_tokens": planner_side_usage.get("completion_tokens"),
        "planner_side_total_tokens": planner_side_usage.get("total_tokens"),
        "total_prompt_tokens": all_stage_usage.get("prompt_tokens"),
        "total_completion_tokens": all_stage_usage.get("completion_tokens"),
        "total_tokens": all_stage_usage.get("total_tokens"),
        "token_usage_stage_count": all_stage_usage.get("stage_count"),
        "token_usage_stages_reported": all_stage_usage.get("stages_with_usage"),
        "token_usage_coverage": all_stage_usage.get("usage_coverage"),
        "selector_estimated_input_cost": selector_stage_metrics.get("estimated_input_cost"),
        "selector_estimated_output_cost": selector_stage_metrics.get("estimated_output_cost"),
        "selector_estimated_total_cost": selector_stage_metrics.get("estimated_total_cost"),
        "planner_estimated_input_cost": planner_stage_metrics.get("estimated_input_cost"),
        "planner_estimated_output_cost": planner_stage_metrics.get("estimated_output_cost"),
        "planner_estimated_total_cost": planner_stage_metrics.get("estimated_total_cost"),
        "executor_estimated_input_cost": executor_stage_metrics.get("estimated_input_cost"),
        "executor_estimated_output_cost": executor_stage_metrics.get("estimated_output_cost"),
        "executor_estimated_total_cost": executor_stage_metrics.get("estimated_total_cost"),
        "executor_side_estimated_input_cost": executor_side_cost.get("estimated_input_cost"),
        "executor_side_estimated_output_cost": executor_side_cost.get("estimated_output_cost"),
        "executor_side_estimated_total_cost": executor_side_cost.get("estimated_total_cost"),
        "planner_side_estimated_input_cost": planner_side_cost.get("estimated_input_cost"),
        "planner_side_estimated_output_cost": planner_side_cost.get("estimated_output_cost"),
        "planner_side_estimated_total_cost": planner_side_cost.get("estimated_total_cost"),
        "estimated_total_input_cost": all_stage_cost.get("estimated_input_cost"),
        "estimated_total_output_cost": all_stage_cost.get("estimated_output_cost"),
        "estimated_total_cost": all_stage_cost.get("estimated_total_cost"),
        "cost_currency": str(metadata.get("cost_currency") or args.cost_currency),
        "expected_keys_json": json.dumps(expected_keys, ensure_ascii=False),
        "hidden_columns_json": json.dumps(hidden_columns, ensure_ascii=False),
        "table_row_count": count_csv_rows(table_copy_path),
        "candidate_row_count": count_csv_rows(candidate_table_path),
        "candidate_column_count": count_csv_columns(candidate_focused_path),
        "ground_truth_row_count": len(item["ground_truth_rows"]),
        "assigned_ground_truth_row_count": row_level_count,
        "predicted_row_count": pred_count,
        "row_selection_recall": row_recall,
        "row_selection_precision": row_precision,
        "row_selection_f1": row_f1,
        "exact_match_rate": exact_match_rate,
        "all_rows_exact_match": all_rows_exact_match,
        "selector_error": selector_error,
        "planner_error": planner_error,
        "executor_error": executor_error,
        "selector_used_fallback_full_table": int(metadata.get("selector_used_fallback_full_table") or 0),
        "planner_plan_json": json.dumps(metadata.get("planner_plan") or {}, ensure_ascii=False),
        "parsed_predictions_json": json.dumps(metadata.get("parsed_predictions") or [], ensure_ascii=False),
        "question_dir": str(question_dir),
        "table_copy_csv": str(table_copy_path),
        "candidate_table_csv": str(candidate_table_path),
        "candidate_table_focused_csv": str(candidate_focused_path),
        "ground_truth_table_csv": str(ground_truth_table_path),
        "ground_truth_with_rowids_csv": str(ground_truth_with_rowids_path),
        "selector_prompt_txt": str(selector_prompt_path),
        "planner_prompt_txt": str(planner_prompt_path),
        "executor_prompt_txt": str(executor_prompt_path),
    }

    return {
        "skip_reason": None,
        "request_rows": request_rows,
        "response_rows": responses_rows,
        "question_row": question_row,
        "row_rows": llm_rows,
    }


def process_question_item(
    *,
    item: Dict[str, Any],
    table_cols: Sequence[str],
    full_table_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    logger,
    selector_prompt_template: str,
    planner_prompt_template: str,
    executor_prompt_template: str,
) -> Dict[str, Any]:
    question = item["question"]
    source_column = item["column_used"]
    if source_column not in table_cols:
        return {
            "skip_reason": "missing_source_column",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
        }

    hidden_columns = choose_hidden_columns(
        annotated_row=item["annotated_row"],
        schema_cols=table_cols,
        source_column=source_column,
        extra_hidden_columns=args.answer_leak_columns,
    )
    visible_cols, visible_rows = drop_columns(full_table_rows, hidden_columns)
    visible_table_csv_text = table_to_csv_text(visible_cols, visible_rows)

    question_slug = sanitize_name(f"{item['item_id']}__{question}")[:180]
    question_dir = run_dir / "questions" / question_slug
    question_dir.mkdir(parents=True, exist_ok=True)

    table_copy_path = question_dir / "table_copy.csv"
    write_csv(table_copy_path, visible_rows)

    assigned_gt_rows = assign_ground_truth_row_ids(
        full_table_rows=visible_rows,
        source_column=source_column,
        ground_truth_rows=item["ground_truth_rows"],
    )
    gt_with_rowids_csv = question_dir / "ground_truth_with_rowids.csv"
    write_csv(gt_with_rowids_csv, assigned_gt_rows)
    write_csv(question_dir / "ground_truth_table.csv", item["ground_truth_rows"])

    request_rows: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []
    executor_endpoint = resolve_stage_endpoint(args, "executor")
    planner_endpoint = resolve_stage_endpoint(args, "planner")

    selector_prompt = render_template(
        selector_prompt_template,
        {
            "question": question,
            "headers_json": json.dumps(visible_cols, ensure_ascii=False),
            "table_csv_text": visible_table_csv_text,
        },
    )
    selector_prompt_chars = len(selector_prompt)
    (question_dir / "selector_prompt.txt").write_text(selector_prompt, encoding="utf-8")
    request_rows.append(
        {
            "stage": "selector",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": executor_endpoint["api_base"],
            "model_name": executor_endpoint["model_name"],
            "prompt": selector_prompt,
            "prompt_char_count": selector_prompt_chars,
        }
    )

    selector_raw_text = ""
    selector_error = ""
    selector_meta: Dict[str, Any] = {}
    selector_output: Dict[str, Any] = {"selected_row_ids": [], "selection_reason": "", "needs_derivation": True}
    try:
        selector_resp = run_one_call_with_retries(
            api_base=executor_endpoint["api_base"],
            api_key=executor_endpoint["api_key"],
            model_name=executor_endpoint["model_name"],
            prompt=selector_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.selector_max_tokens,
            timeout=args.timeout,
            num_retries=args.num_retries,
            logger=logger,
        )
        selector_raw_text = parse_chat_completion_text(selector_resp)
        selector_meta = completion_meta(selector_resp)
        selector_output = parse_selector_output(selector_raw_text)
    except Exception as exc:
        selector_error = str(exc)
        selector_output = {
            "selected_row_ids": [],
            "selection_reason": "Selector failed; using full visible table as fallback.",
            "needs_derivation": True,
        }
    selector_usage = extract_usage_metrics(selector_meta)
    selector_cost = estimate_token_cost(
        prompt_tokens=selector_usage["prompt_tokens"],
        completion_tokens=selector_usage["completion_tokens"],
        **stage_pricing(args, "selector"),
    )

    selected_rows = filter_rows_by_ids(visible_rows, selector_output.get("selected_row_ids") or [])
    selector_used_fallback = False
    if not selected_rows:
        selected_rows = list(visible_rows)
        selector_used_fallback = True
    candidate_table_path = question_dir / "candidate_table.csv"
    write_csv(candidate_table_path, selected_rows)
    response_rows.append(
        {
            "stage": "selector",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "llm_raw_output": selector_raw_text,
            "parsed_output_json": json.dumps(selector_output, ensure_ascii=False),
            "error": selector_error,
            "prompt_char_count": selector_prompt_chars,
            "prompt_tokens": selector_usage["prompt_tokens"],
            "completion_tokens": selector_usage["completion_tokens"],
            "total_tokens": selector_usage["total_tokens"],
            "usage_available": selector_usage["usage_available"],
            "estimated_input_cost": selector_cost["estimated_input_cost"],
            "estimated_output_cost": selector_cost["estimated_output_cost"],
            "estimated_total_cost": selector_cost["estimated_total_cost"],
            "cost_currency": args.cost_currency,
            "model_meta_json": json.dumps(selector_meta, ensure_ascii=False),
        }
    )
    write_json(
        question_dir / "selector_response.json",
        {
            "raw_text": selector_raw_text,
            "parsed_output": selector_output,
            "error": selector_error,
            "used_fallback_full_table": selector_used_fallback,
            "prompt_char_count": selector_prompt_chars,
            "usage": selector_usage,
            "estimated_cost": selector_cost,
            "cost_currency": args.cost_currency,
            "model_meta": selector_meta,
        },
    )

    planner_prompt = render_template(
        planner_prompt_template,
        {
            "overall_context": (
                "You are a hybrid querying planner for clinical-trials questions. "
                "Your job is to understand the essence of what the question is asking, "
                "identify which visible information matters, and describe how another "
                "model should derive the answer for each selected row from visible "
                "values and row context alone. You must infer which visible column is "
                "most likely the source evidence column from the question and visible "
                "headers themselves. Do not assume the answer already exists verbatim "
                "in the table."
            ),
            "question": question,
            "headers_json": json.dumps(visible_cols, ensure_ascii=False),
            "candidate_row_count": len(selected_rows),
            "candidate_preview_json": candidate_preview_json(
                selected_rows,
                visible_cols=visible_cols,
                max_rows=max(1, int(args.planner_preview_rows)),
            ),
        },
    )
    planner_prompt_chars = len(planner_prompt)
    (question_dir / "planner_prompt.txt").write_text(planner_prompt, encoding="utf-8")
    request_rows.append(
        {
            "stage": "planner",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": planner_endpoint["api_base"],
            "model_name": planner_endpoint["model_name"],
            "prompt": planner_prompt,
            "prompt_char_count": planner_prompt_chars,
        }
    )
    planner_raw_text = ""
    planner_error = ""
    planner_meta: Dict[str, Any] = {}
    planner_plan = build_fallback_plan(question=question, visible_cols=visible_cols)
    try:
        planner_resp = run_one_call_with_retries(
            api_base=planner_endpoint["api_base"],
            api_key=planner_endpoint["api_key"],
            model_name=planner_endpoint["model_name"],
            prompt=planner_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.planner_max_tokens,
            timeout=args.timeout,
            num_retries=args.num_retries,
            logger=logger,
        )
        planner_raw_text = parse_chat_completion_text(planner_resp)
        planner_meta = completion_meta(planner_resp)
        planner_plan = parse_planner_output(
            planner_raw_text,
            question=question,
            visible_cols=visible_cols,
        )
    except Exception as exc:
        planner_error = str(exc)
    planner_usage = extract_usage_metrics(planner_meta)
    planner_cost = estimate_token_cost(
        prompt_tokens=planner_usage["prompt_tokens"],
        completion_tokens=planner_usage["completion_tokens"],
        **stage_pricing(args, "planner"),
    )

    response_rows.append(
        {
            "stage": "planner",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "llm_raw_output": planner_raw_text,
            "parsed_output_json": json.dumps(planner_plan, ensure_ascii=False),
            "error": planner_error,
            "prompt_char_count": planner_prompt_chars,
            "prompt_tokens": planner_usage["prompt_tokens"],
            "completion_tokens": planner_usage["completion_tokens"],
            "total_tokens": planner_usage["total_tokens"],
            "usage_available": planner_usage["usage_available"],
            "estimated_input_cost": planner_cost["estimated_input_cost"],
            "estimated_output_cost": planner_cost["estimated_output_cost"],
            "estimated_total_cost": planner_cost["estimated_total_cost"],
            "cost_currency": args.cost_currency,
            "model_meta_json": json.dumps(planner_meta, ensure_ascii=False),
        }
    )
    write_json(
        question_dir / "planner_response.json",
        {
            "raw_text": planner_raw_text,
            "parsed_plan": planner_plan,
            "error": planner_error,
            "prompt_char_count": planner_prompt_chars,
            "usage": planner_usage,
            "estimated_cost": planner_cost,
            "cost_currency": args.cost_currency,
            "model_meta": planner_meta,
        },
    )

    candidate_cols = planner_plan.get("relevant_columns") or visible_cols
    candidate_rows = project_rows(selected_rows, candidate_cols)
    candidate_focused_table_path = question_dir / "candidate_table_focused.csv"
    write_csv(candidate_focused_table_path, candidate_rows)

    executor_prompt = render_template(
        executor_prompt_template,
        {
            "question": question,
            "planner_plan_json": json.dumps(planner_plan, ensure_ascii=False, indent=2),
            "candidate_table_csv_text": table_to_csv_text(candidate_cols, candidate_rows),
        },
    )
    executor_prompt_chars = len(executor_prompt)
    (question_dir / "executor_prompt.txt").write_text(executor_prompt, encoding="utf-8")
    request_rows.append(
        {
            "stage": "executor_final",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "api_base": executor_endpoint["api_base"],
            "model_name": executor_endpoint["model_name"],
            "prompt": executor_prompt,
            "prompt_char_count": executor_prompt_chars,
        }
    )

    final_raw_text = ""
    final_error = ""
    final_meta: Dict[str, Any] = {}
    parsed_predictions: List[Dict[str, Any]] = []
    try:
        final_resp = run_one_call_with_retries(
            api_base=executor_endpoint["api_base"],
            api_key=executor_endpoint["api_key"],
            model_name=executor_endpoint["model_name"],
            prompt=executor_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.executor_max_tokens,
            timeout=args.timeout,
            num_retries=args.num_retries,
            logger=logger,
        )
        final_raw_text = parse_chat_completion_text(final_resp)
        final_meta = completion_meta(final_resp)
        parsed_predictions = parse_final_predictions(final_raw_text)
    except Exception as exc:
        final_error = str(exc)
    executor_usage = extract_usage_metrics(final_meta)
    executor_cost = estimate_token_cost(
        prompt_tokens=executor_usage["prompt_tokens"],
        completion_tokens=executor_usage["completion_tokens"],
        **stage_pricing(args, "executor_final"),
    )

    response_rows.append(
        {
            "stage": "executor_final",
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "llm_raw_output": final_raw_text,
            "parsed_output_json": json.dumps(parsed_predictions, ensure_ascii=False),
            "error": final_error,
            "prompt_char_count": executor_prompt_chars,
            "prompt_tokens": executor_usage["prompt_tokens"],
            "completion_tokens": executor_usage["completion_tokens"],
            "total_tokens": executor_usage["total_tokens"],
            "usage_available": executor_usage["usage_available"],
            "estimated_input_cost": executor_cost["estimated_input_cost"],
            "estimated_output_cost": executor_cost["estimated_output_cost"],
            "estimated_total_cost": executor_cost["estimated_total_cost"],
            "cost_currency": args.cost_currency,
            "model_meta_json": json.dumps(final_meta, ensure_ascii=False),
        }
    )
    write_json(
        question_dir / "executor_response.json",
        {
            "raw_text": final_raw_text,
            "parsed_predictions": parsed_predictions,
            "error": final_error,
            "prompt_char_count": executor_prompt_chars,
            "usage": executor_usage,
            "estimated_cost": executor_cost,
            "cost_currency": args.cost_currency,
            "model_meta": final_meta,
        },
    )

    selector_stage_metrics = {
        "prompt_chars": selector_prompt_chars,
        **selector_usage,
        **selector_cost,
    }
    planner_stage_metrics = {
        "prompt_chars": planner_prompt_chars,
        **planner_usage,
        **planner_cost,
    }
    executor_stage_metrics = {
        "prompt_chars": executor_prompt_chars,
        **executor_usage,
        **executor_cost,
    }
    executor_side_usage = combine_usage_metrics([selector_stage_metrics, executor_stage_metrics])
    planner_side_usage = combine_usage_metrics([planner_stage_metrics])
    all_stage_usage = combine_usage_metrics([selector_stage_metrics, planner_stage_metrics, executor_stage_metrics])
    executor_side_cost = combine_cost_metrics([selector_stage_metrics, executor_stage_metrics])
    planner_side_cost = combine_cost_metrics([planner_stage_metrics])
    all_stage_cost = combine_cost_metrics([selector_stage_metrics, planner_stage_metrics, executor_stage_metrics])

    pred_map: Dict[int, Any] = {}
    for pred in parsed_predictions:
        row_id = pred["table_row_id"]
        if row_id not in pred_map:
            pred_map[row_id] = pred["answer"]

    gt_row_ids = [row.get("table_row_id") for row in assigned_gt_rows if row.get("table_row_id") is not None]
    gt_row_id_set = {int(row_id) for row_id in gt_row_ids}
    pred_row_id_set = set(pred_map.keys())
    matched_row_ids = gt_row_id_set & pred_row_id_set

    matched_exact = 0
    row_level_count = 0
    row_output_rows: List[Dict[str, Any]] = []
    for gt_row in assigned_gt_rows:
        table_row_id = gt_row.get("table_row_id")
        actual_payload = parse_jsonish(gt_row.get("derived_expected_llm_response", ""))
        predicted_answer = pred_map.get(table_row_id) if table_row_id is not None else None
        actual_values = extract_payload_values(gt_row.get("derived_expected_llm_response", ""))
        exact = exact_match(predicted_answer, actual_payload) if table_row_id is not None else 0.0
        any_value = value_match(predicted_answer, actual_values) if table_row_id is not None else 0.0
        if exact >= 1.0 - 1e-12:
            matched_exact += 1
        if table_row_id is not None:
            row_level_count += 1
        row_output_rows.append(
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "table_row_id": table_row_id,
                "NCT": gt_row.get("NCT", ""),
                "PubMed ID": gt_row.get("PubMed ID", ""),
                "Trial name": gt_row.get("Trial name", ""),
                "source_value": gt_row.get("source_value", ""),
                "actual_payload_json": json.dumps(canonical_jsonable(actual_payload), ensure_ascii=False, sort_keys=True),
                "predicted_answer_json": json.dumps(canonical_jsonable(predicted_answer), ensure_ascii=False, sort_keys=True),
                "actual_values_json": json.dumps(actual_values, ensure_ascii=False),
                "row_selected": int(table_row_id in pred_row_id_set) if table_row_id is not None else 0,
                "exact_match": exact,
                "match_any_value": any_value,
                "question_dir": str(question_dir),
            }
        )

    extra_predictions = sorted(pred_row_id_set - gt_row_id_set)
    for row_id in extra_predictions:
        row_output_rows.append(
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "table_row_id": row_id,
                "NCT": "",
                "PubMed ID": "",
                "Trial name": "",
                "source_value": "",
                "actual_payload_json": "",
                "predicted_answer_json": json.dumps(canonical_jsonable(pred_map.get(row_id)), ensure_ascii=False, sort_keys=True),
                "actual_values_json": "[]",
                "row_selected": 0,
                "exact_match": 0.0,
                "match_any_value": 0.0,
                "question_dir": str(question_dir),
                "row_type": "extra_prediction",
            }
        )

    gt_count = len(gt_row_id_set)
    pred_count = len(pred_row_id_set)
    exact_match_rate = (matched_exact / row_level_count) if row_level_count else 0.0
    row_recall = (len(matched_row_ids) / gt_count) if gt_count else (1.0 if pred_count == 0 else 0.0)
    row_precision = (len(matched_row_ids) / pred_count) if pred_count else (1.0 if gt_count == 0 else 0.0)
    if row_precision + row_recall > 0:
        row_f1 = 2.0 * row_precision * row_recall / (row_precision + row_recall)
    else:
        row_f1 = 0.0
    all_rows_exact_match = 1.0 if gt_count == pred_count and matched_exact == row_level_count and gt_count == len(matched_row_ids) else 0.0

    question_row = {
        "item_id": item["item_id"],
        "csv_row_number": item["csv_row_number"],
        "question": question,
        "column_used": source_column,
        "planner_inferred_source_column": str(planner_plan.get("inferred_source_column") or ""),
        "planner_inferred_source_matches_metadata": int(
            str(planner_plan.get("inferred_source_column") or "") == str(source_column)
        ),
        "selector_prompt_chars": selector_prompt_chars,
        "planner_prompt_chars": planner_prompt_chars,
        "executor_prompt_chars": executor_prompt_chars,
        "executor_side_prompt_chars": executor_side_usage["prompt_chars"],
        "total_prompt_chars": all_stage_usage["prompt_chars"],
        "selector_prompt_tokens": selector_usage["prompt_tokens"],
        "selector_completion_tokens": selector_usage["completion_tokens"],
        "selector_total_tokens": selector_usage["total_tokens"],
        "planner_prompt_tokens": planner_usage["prompt_tokens"],
        "planner_completion_tokens": planner_usage["completion_tokens"],
        "planner_total_tokens": planner_usage["total_tokens"],
        "executor_prompt_tokens": executor_usage["prompt_tokens"],
        "executor_completion_tokens": executor_usage["completion_tokens"],
        "executor_total_tokens": executor_usage["total_tokens"],
        "executor_side_prompt_tokens": executor_side_usage["prompt_tokens"],
        "executor_side_completion_tokens": executor_side_usage["completion_tokens"],
        "executor_side_total_tokens": executor_side_usage["total_tokens"],
        "planner_side_prompt_tokens": planner_side_usage["prompt_tokens"],
        "planner_side_completion_tokens": planner_side_usage["completion_tokens"],
        "planner_side_total_tokens": planner_side_usage["total_tokens"],
        "total_prompt_tokens": all_stage_usage["prompt_tokens"],
        "total_completion_tokens": all_stage_usage["completion_tokens"],
        "total_tokens": all_stage_usage["total_tokens"],
        "token_usage_stage_count": all_stage_usage["stage_count"],
        "token_usage_stages_reported": all_stage_usage["stages_with_usage"],
        "token_usage_coverage": all_stage_usage["usage_coverage"],
        "selector_estimated_input_cost": selector_cost["estimated_input_cost"],
        "selector_estimated_output_cost": selector_cost["estimated_output_cost"],
        "selector_estimated_total_cost": selector_cost["estimated_total_cost"],
        "planner_estimated_input_cost": planner_cost["estimated_input_cost"],
        "planner_estimated_output_cost": planner_cost["estimated_output_cost"],
        "planner_estimated_total_cost": planner_cost["estimated_total_cost"],
        "executor_estimated_input_cost": executor_cost["estimated_input_cost"],
        "executor_estimated_output_cost": executor_cost["estimated_output_cost"],
        "executor_estimated_total_cost": executor_cost["estimated_total_cost"],
        "executor_side_estimated_input_cost": executor_side_cost["estimated_input_cost"],
        "executor_side_estimated_output_cost": executor_side_cost["estimated_output_cost"],
        "executor_side_estimated_total_cost": executor_side_cost["estimated_total_cost"],
        "planner_side_estimated_input_cost": planner_side_cost["estimated_input_cost"],
        "planner_side_estimated_output_cost": planner_side_cost["estimated_output_cost"],
        "planner_side_estimated_total_cost": planner_side_cost["estimated_total_cost"],
        "estimated_total_input_cost": all_stage_cost["estimated_input_cost"],
        "estimated_total_output_cost": all_stage_cost["estimated_output_cost"],
        "estimated_total_cost": all_stage_cost["estimated_total_cost"],
        "cost_currency": args.cost_currency,
        "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
        "hidden_columns_json": json.dumps(hidden_columns, ensure_ascii=False),
        "table_row_count": len(visible_rows),
        "candidate_row_count": len(selected_rows),
        "candidate_column_count": len(candidate_cols),
        "ground_truth_row_count": len(item["ground_truth_rows"]),
        "assigned_ground_truth_row_count": row_level_count,
        "predicted_row_count": pred_count,
        "row_selection_recall": row_recall,
        "row_selection_precision": row_precision,
        "row_selection_f1": row_f1,
        "exact_match_rate": exact_match_rate,
        "all_rows_exact_match": all_rows_exact_match,
        "selector_error": selector_error,
        "planner_error": planner_error,
        "executor_error": final_error,
        "selector_used_fallback_full_table": int(selector_used_fallback),
        "planner_plan_json": json.dumps(planner_plan, ensure_ascii=False),
        "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
        "question_dir": str(question_dir),
        "table_copy_csv": str(table_copy_path),
        "candidate_table_csv": str(candidate_table_path),
        "candidate_table_focused_csv": str(candidate_focused_table_path),
        "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
        "ground_truth_with_rowids_csv": str(gt_with_rowids_csv),
        "selector_prompt_txt": str(question_dir / "selector_prompt.txt"),
        "planner_prompt_txt": str(question_dir / "planner_prompt.txt"),
        "executor_prompt_txt": str(question_dir / "executor_prompt.txt"),
    }

    write_csv(question_dir / "llm_predictions.csv", row_output_rows)
    write_json(
        question_dir / "metadata.json",
        {
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "column_used": source_column,
            "planner_inferred_source_column": str(planner_plan.get("inferred_source_column") or ""),
            "planner_inferred_source_matches_metadata": int(
                str(planner_plan.get("inferred_source_column") or "") == str(source_column)
            ),
            "selector_stage_usage": selector_stage_metrics,
            "planner_stage_usage": planner_stage_metrics,
            "executor_stage_usage": executor_stage_metrics,
            "executor_side_usage": executor_side_usage,
            "planner_side_usage": planner_side_usage,
            "all_stage_usage": all_stage_usage,
            "executor_side_estimated_cost": executor_side_cost,
            "planner_side_estimated_cost": planner_side_cost,
            "all_stage_estimated_cost": all_stage_cost,
            "cost_currency": args.cost_currency,
            "expected_keys": item["expected_keys"],
            "hidden_columns": hidden_columns,
            "selector_used_fallback_full_table": selector_used_fallback,
            "table_copy_csv": str(table_copy_path),
            "candidate_table_csv": str(candidate_table_path),
            "candidate_table_focused_csv": str(candidate_focused_table_path),
            "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
            "ground_truth_with_rowids_csv": str(gt_with_rowids_csv),
            "selector_prompt_txt": str(question_dir / "selector_prompt.txt"),
            "planner_prompt_txt": str(question_dir / "planner_prompt.txt"),
            "executor_prompt_txt": str(question_dir / "executor_prompt.txt"),
            "selector_output": selector_output,
            "planner_plan": planner_plan,
            "parsed_predictions": parsed_predictions,
            "selector_error": selector_error,
            "planner_error": planner_error,
            "executor_error": final_error,
            "executor_endpoint": executor_endpoint,
            "planner_endpoint": planner_endpoint,
            "selector_model_meta": selector_meta,
            "planner_model_meta": planner_meta,
            "executor_model_meta": final_meta,
        },
    )

    return {
        "skip_reason": None,
        "request_rows": request_rows,
        "response_rows": response_rows,
        "question_row": question_row,
        "row_rows": row_output_rows,
    }


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    annotated_csv = Path(args.annotated_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    selector_prompt_file = Path(args.selector_prompt_file).expanduser().resolve()
    planner_prompt_file = Path(args.planner_prompt_file).expanduser().resolve()
    executor_prompt_file = Path(args.executor_prompt_file).expanduser().resolve()

    run_dir = make_run_dir(run_root, args.run_name.strip() or "question_table_copy_llm_planner_executor")
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"qtable_planexec_{run_dir.name}")

    logger.info("Manifest CSV: %s", manifest_csv)
    logger.info("Annotated CSV: %s", annotated_csv)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Executor model: %s", args.executor_model_name or args.model_name)
    logger.info("Executor API base: %s", args.executor_api_base or args.api_base)
    logger.info("Planner model: %s", args.planner_model_name or args.executor_model_name or args.model_name)
    logger.info("Planner API base: %s", args.planner_api_base or args.api_base)
    logger.info("Max in flight: %s", args.max_in_flight)

    selector_prompt_template = load_text(selector_prompt_file)
    planner_prompt_template = load_text(planner_prompt_file)
    executor_prompt_template = load_text(executor_prompt_file)

    sample_questions = load_sample_questions(
        manifest_csv=manifest_csv,
        annotated_csv=annotated_csv,
        limit=args.limit,
        csv_row_number=args.csv_row_number,
    )
    sample_manifest_rows = [
        {
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": item["question"],
            "column_used": item["column_used"],
            "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
            "question_dir": item["question_dir"],
            "ground_truth_row_count": len(item["ground_truth_rows"]),
        }
        for item in sample_questions
    ]
    write_csv(run_dir / "sample_manifest.csv", sample_manifest_rows)

    pending_questions: List[Dict[str, Any]] = []

    conn = sqlite3.connect(db_path)
    try:
        table_cols, full_table_rows = fetch_full_table(conn, args.table_name)
    finally:
        conn.close()

    question_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []
    requests_rows: List[Dict[str, Any]] = []
    responses_rows: List[Dict[str, Any]] = []
    run_counter = Counter()

    def question_had_error(result: Dict[str, Any]) -> bool:
        if result.get("skip_reason"):
            return True
        question_row = result.get("question_row") or {}
        return bool(
            question_row.get("selector_error")
            or question_row.get("planner_error")
            or question_row.get("executor_error")
        )

    if args.resume:
        for item in sample_questions:
            resumed_result = load_resumed_question_result(
                item=item,
                run_dir=run_dir,
                args=args,
            )
            if resumed_result is None:
                pending_questions.append(item)
                continue
            requests_rows.extend(resumed_result["request_rows"])
            responses_rows.extend(resumed_result["response_rows"])
            question_rows.append(resumed_result["question_row"])
            row_rows.extend(resumed_result["row_rows"])
            run_counter["resumed_completed"] += 1
    else:
        pending_questions = list(sample_questions)

    logger.info(
        "Resume enabled: %s | resumed_completed=%d | pending_questions=%d",
        int(bool(args.resume)),
        int(run_counter.get("resumed_completed", 0)),
        len(pending_questions),
    )

    progress_total = len(sample_questions)
    progress_completed = len(question_rows)
    progress_success = sum(1 for row in question_rows if not question_row_has_errors(row))
    progress_error = progress_completed - progress_success
    progress_start_time = time.time()
    if progress_total:
        print_progress(
            label="Questions",
            completed=progress_completed,
            total=progress_total,
            success=progress_success,
            error=progress_error,
            start_time=progress_start_time,
        )

    max_workers = max(1, int(args.max_in_flight))
    if max_workers == 1:
        for item in pending_questions:
            result = process_question_item(
                item=item,
                table_cols=table_cols,
                full_table_rows=full_table_rows,
                args=args,
                run_dir=run_dir,
                logger=logger,
                selector_prompt_template=selector_prompt_template,
                planner_prompt_template=planner_prompt_template,
                executor_prompt_template=executor_prompt_template,
            )
            progress_completed += 1
            if question_had_error(result):
                progress_error += 1
            else:
                progress_success += 1
            skip_reason = result.get("skip_reason")
            if skip_reason:
                run_counter[skip_reason] += 1
            else:
                requests_rows.extend(result["request_rows"])
                responses_rows.extend(result["response_rows"])
                question_rows.append(result["question_row"])
                row_rows.extend(result["row_rows"])
            print_progress(
                label="Questions",
                completed=progress_completed,
                total=progress_total,
                success=progress_success,
                error=progress_error,
                start_time=progress_start_time,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    process_question_item,
                    item=item,
                    table_cols=table_cols,
                    full_table_rows=full_table_rows,
                    args=args,
                    run_dir=run_dir,
                    logger=logger,
                    selector_prompt_template=selector_prompt_template,
                    planner_prompt_template=planner_prompt_template,
                    executor_prompt_template=executor_prompt_template,
                ): item
                for item in pending_questions
            }
            for future in as_completed(future_map):
                result = future.result()
                progress_completed += 1
                if question_had_error(result):
                    progress_error += 1
                else:
                    progress_success += 1
                skip_reason = result.get("skip_reason")
                if skip_reason:
                    run_counter[skip_reason] += 1
                else:
                    requests_rows.extend(result["request_rows"])
                    responses_rows.extend(result["response_rows"])
                    question_rows.append(result["question_row"])
                    row_rows.extend(result["row_rows"])
                print_progress(
                    label="Questions",
                    completed=progress_completed,
                    total=progress_total,
                    success=progress_success,
                    error=progress_error,
                    start_time=progress_start_time,
                )

    question_rows.sort(key=lambda row: int(row.get("csv_row_number") or 0))
    requests_rows.sort(key=lambda row: (int(row.get("csv_row_number") or 0), str(row.get("stage") or "")))
    responses_rows.sort(key=lambda row: (int(row.get("csv_row_number") or 0), str(row.get("stage") or "")))
    row_rows.sort(
        key=lambda row: (
            int(row.get("csv_row_number") or 0),
            int(row.get("table_row_id") or -1) if str(row.get("table_row_id", "")).strip() else -1,
            str(row.get("row_type") or ""),
        )
    )

    question_results_csv = run_dir / "all_question_results.csv"
    row_level_csv = run_dir / "row_level_predictions.csv"
    requests_csv = run_dir / "requests.csv"
    responses_csv = run_dir / "responses.csv"
    write_csv(question_results_csv, question_rows)
    write_csv(row_level_csv, row_rows)
    write_csv(requests_csv, requests_rows)
    write_csv(responses_csv, responses_rows)

    summary_rows = [
        {
            "question_count": len(question_rows),
            "avg_candidate_row_count": avg_numeric(question_rows, "candidate_row_count"),
            "avg_row_selection_recall": avg_numeric(question_rows, "row_selection_recall"),
            "avg_row_selection_precision": avg_numeric(question_rows, "row_selection_precision"),
            "avg_row_selection_f1": avg_numeric(question_rows, "row_selection_f1"),
            "avg_exact_match_rate": avg_numeric(question_rows, "exact_match_rate"),
            "avg_all_rows_exact_match": avg_numeric(question_rows, "all_rows_exact_match"),
            "selector_fallback_rate": avg_numeric(question_rows, "selector_used_fallback_full_table"),
            "avg_selector_prompt_chars": avg_numeric(question_rows, "selector_prompt_chars"),
            "avg_planner_prompt_chars": avg_numeric(question_rows, "planner_prompt_chars"),
            "avg_executor_prompt_chars": avg_numeric(question_rows, "executor_prompt_chars"),
            "avg_total_prompt_chars": avg_numeric(question_rows, "total_prompt_chars"),
            "avg_selector_total_tokens": avg_numeric(question_rows, "selector_total_tokens"),
            "avg_planner_total_tokens": avg_numeric(question_rows, "planner_total_tokens"),
            "avg_executor_total_tokens": avg_numeric(question_rows, "executor_total_tokens"),
            "avg_executor_side_total_tokens": avg_numeric(question_rows, "executor_side_total_tokens"),
            "avg_planner_side_total_tokens": avg_numeric(question_rows, "planner_side_total_tokens"),
            "avg_total_tokens": avg_numeric(question_rows, "total_tokens"),
            "avg_total_prompt_tokens": avg_numeric(question_rows, "total_prompt_tokens"),
            "avg_total_completion_tokens": avg_numeric(question_rows, "total_completion_tokens"),
            "avg_token_usage_coverage": avg_numeric(question_rows, "token_usage_coverage"),
            "avg_selector_estimated_total_cost": avg_numeric_or_none(question_rows, "selector_estimated_total_cost"),
            "avg_planner_estimated_total_cost": avg_numeric_or_none(question_rows, "planner_estimated_total_cost"),
            "avg_executor_estimated_total_cost": avg_numeric_or_none(question_rows, "executor_estimated_total_cost"),
            "avg_executor_side_estimated_total_cost": avg_numeric_or_none(question_rows, "executor_side_estimated_total_cost"),
            "avg_planner_side_estimated_total_cost": avg_numeric_or_none(question_rows, "planner_side_estimated_total_cost"),
            "avg_estimated_total_cost": avg_numeric_or_none(question_rows, "estimated_total_cost"),
            "cost_currency": args.cost_currency,
        }
    ]
    baseline_summary_csv = run_dir / "baseline_summary.csv"
    write_csv(baseline_summary_csv, summary_rows)

    token_cost_summary_rows = [
        {
            "question_count": len(question_rows),
            "questions_with_any_usage": int(sum(1 for row in question_rows if float(row.get("token_usage_coverage") or 0.0) > 0.0)),
            "questions_with_full_usage": int(sum(1 for row in question_rows if float(row.get("token_usage_coverage") or 0.0) >= 1.0)),
            "total_selector_prompt_chars": sum_numeric(question_rows, "selector_prompt_chars"),
            "total_planner_prompt_chars": sum_numeric(question_rows, "planner_prompt_chars"),
            "total_executor_prompt_chars": sum_numeric(question_rows, "executor_prompt_chars"),
            "total_prompt_chars": sum_numeric(question_rows, "total_prompt_chars"),
            "total_selector_prompt_tokens": sum_numeric(question_rows, "selector_prompt_tokens"),
            "total_selector_completion_tokens": sum_numeric(question_rows, "selector_completion_tokens"),
            "total_selector_tokens": sum_numeric(question_rows, "selector_total_tokens"),
            "total_planner_prompt_tokens": sum_numeric(question_rows, "planner_prompt_tokens"),
            "total_planner_completion_tokens": sum_numeric(question_rows, "planner_completion_tokens"),
            "total_planner_tokens": sum_numeric(question_rows, "planner_total_tokens"),
            "total_executor_prompt_tokens": sum_numeric(question_rows, "executor_prompt_tokens"),
            "total_executor_completion_tokens": sum_numeric(question_rows, "executor_completion_tokens"),
            "total_executor_tokens": sum_numeric(question_rows, "executor_total_tokens"),
            "total_executor_side_prompt_tokens": sum_numeric(question_rows, "executor_side_prompt_tokens"),
            "total_executor_side_completion_tokens": sum_numeric(question_rows, "executor_side_completion_tokens"),
            "total_executor_side_tokens": sum_numeric(question_rows, "executor_side_total_tokens"),
            "total_planner_side_prompt_tokens": sum_numeric(question_rows, "planner_side_prompt_tokens"),
            "total_planner_side_completion_tokens": sum_numeric(question_rows, "planner_side_completion_tokens"),
            "total_planner_side_tokens": sum_numeric(question_rows, "planner_side_total_tokens"),
            "total_prompt_tokens": sum_numeric(question_rows, "total_prompt_tokens"),
            "total_completion_tokens": sum_numeric(question_rows, "total_completion_tokens"),
            "total_tokens": sum_numeric(question_rows, "total_tokens"),
            "avg_token_usage_coverage": avg_numeric(question_rows, "token_usage_coverage"),
            "total_selector_estimated_input_cost": sum_numeric_or_none(question_rows, "selector_estimated_input_cost"),
            "total_selector_estimated_output_cost": sum_numeric_or_none(question_rows, "selector_estimated_output_cost"),
            "total_selector_estimated_cost": sum_numeric_or_none(question_rows, "selector_estimated_total_cost"),
            "total_planner_estimated_input_cost": sum_numeric_or_none(question_rows, "planner_estimated_input_cost"),
            "total_planner_estimated_output_cost": sum_numeric_or_none(question_rows, "planner_estimated_output_cost"),
            "total_planner_estimated_cost": sum_numeric_or_none(question_rows, "planner_estimated_total_cost"),
            "total_executor_estimated_input_cost": sum_numeric_or_none(question_rows, "executor_estimated_input_cost"),
            "total_executor_estimated_output_cost": sum_numeric_or_none(question_rows, "executor_estimated_output_cost"),
            "total_executor_estimated_cost": sum_numeric_or_none(question_rows, "executor_estimated_total_cost"),
            "total_executor_side_estimated_input_cost": sum_numeric_or_none(question_rows, "executor_side_estimated_input_cost"),
            "total_executor_side_estimated_output_cost": sum_numeric_or_none(question_rows, "executor_side_estimated_output_cost"),
            "total_executor_side_estimated_cost": sum_numeric_or_none(question_rows, "executor_side_estimated_total_cost"),
            "total_planner_side_estimated_input_cost": sum_numeric_or_none(question_rows, "planner_side_estimated_input_cost"),
            "total_planner_side_estimated_output_cost": sum_numeric_or_none(question_rows, "planner_side_estimated_output_cost"),
            "total_planner_side_estimated_cost": sum_numeric_or_none(question_rows, "planner_side_estimated_total_cost"),
            "total_estimated_input_cost": sum_numeric_or_none(question_rows, "estimated_total_input_cost"),
            "total_estimated_output_cost": sum_numeric_or_none(question_rows, "estimated_total_output_cost"),
            "total_estimated_cost": sum_numeric_or_none(question_rows, "estimated_total_cost"),
            "avg_estimated_total_cost": avg_numeric_or_none(question_rows, "estimated_total_cost"),
            "cost_currency": args.cost_currency,
        }
    ]
    token_cost_summary_csv = run_dir / "token_cost_summary.csv"
    write_csv(token_cost_summary_csv, token_cost_summary_rows)

    summary = {
        "manifest_csv": str(manifest_csv),
        "annotated_csv": str(annotated_csv),
        "db_path": str(db_path),
        "executor_model_name": args.executor_model_name or args.model_name,
        "executor_api_base": args.executor_api_base or args.api_base,
        "planner_model_name": args.planner_model_name or args.executor_model_name or args.model_name,
        "planner_api_base": args.planner_api_base or args.api_base,
        "sample_question_count": len(sample_questions),
        "question_result_count": len(question_rows),
        "row_result_count": len(row_rows),
        "cost_currency": args.cost_currency,
        "planner_input_cost_per_million_tokens": float(args.planner_input_cost_per_million_tokens),
        "planner_output_cost_per_million_tokens": float(args.planner_output_cost_per_million_tokens),
        "executor_input_cost_per_million_tokens": float(args.executor_input_cost_per_million_tokens),
        "executor_output_cost_per_million_tokens": float(args.executor_output_cost_per_million_tokens),
        "skips": dict(run_counter),
        "token_cost_summary": token_cost_summary_rows[0],
        "outputs": {
            "sample_manifest_csv": str(run_dir / "sample_manifest.csv"),
            "question_results_csv": str(question_results_csv),
            "row_level_predictions_csv": str(row_level_csv),
            "requests_csv": str(requests_csv),
            "responses_csv": str(responses_csv),
            "baseline_summary_csv": str(baseline_summary_csv),
            "token_cost_summary_csv": str(token_cost_summary_csv),
        },
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote sample manifest: %s", run_dir / "sample_manifest.csv")
    logger.info("Wrote question results: %s", question_results_csv)
    logger.info("Wrote row-level predictions: %s", row_level_csv)
    logger.info("Wrote token/cost summary: %s", token_cost_summary_csv)
    logger.info("Wrote summary: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
