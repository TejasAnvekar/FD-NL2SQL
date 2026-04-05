#!/usr/bin/env python3
"""Generate evidence SQL on a reduced schema, execute it, then infer held-out values."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urllib_request

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_hidden_column_sql_eval import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_RUN_ROOT,
    create_reduced_db,
    extract_sql_candidate,
    fetch_schema,
    load_csv_rows,
    make_run_dir,
    render_schema_hints,
    sanitize_name,
)
from clinicaltrials_api import fetch_study_by_nct, summarize_study  # noqa: E402
from ctgov_hybrid_planner import (  # noqa: E402
    build_ctgov_planning_prompt,
    default_ctgov_plan,
    parse_ctgov_plan,
    select_ctgov_evidence_for_prompt,
)
from question_table_exports import export_question_tables  # noqa: E402
from utils import (  # noqa: E402
    append_jsonl,
    fetch_schema_value_hints,
    is_retryable_provider_error,
    quote_ident,
    setup_logger,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "For each eligible question, remove final_column from the schema, ask a local "
            "vLLM server to generate evidence SQL, execute that SQL, then ask the model to "
            "infer the hidden target values from the executed rows."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--gt_sql_key", default="sql_query")
    ap.add_argument("--column_used_key", default="column_used")
    ap.add_argument("--final_column_key", default="final_column")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--csv_row_number", type=int, default=0, help="0 means all rows; otherwise run only this CSV row number.")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--require_distinct_source_target", type=int, default=1)
    ap.add_argument("--prompt_source_column_mode", choices=("shown", "hidden"), default="hidden")
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="gemma-3-4b-it")
    ap.add_argument("--sql_temperature", type=float, default=0.0)
    ap.add_argument("--sql_top_p", type=float, default=1.0)
    ap.add_argument("--sql_max_tokens", type=int, default=256)
    ap.add_argument("--infer_temperature", type=float, default=0.0)
    ap.add_argument("--infer_top_p", type=float, default=1.0)
    ap.add_argument("--infer_max_tokens", type=int, default=768)
    ap.add_argument("--planner_temperature", type=float, default=0.0)
    ap.add_argument("--planner_top_p", type=float, default=1.0)
    ap.add_argument("--planner_max_tokens", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument("--max_prompt_rows", type=int, default=50)
    ap.add_argument("--use_ctgov_metadata", type=int, default=1)
    ap.add_argument("--use_ctgov_hybrid_planner", type=int, default=1)
    ap.add_argument("--ctgov_timeout", type=float, default=60.0)
    ap.add_argument("--ctgov_max_studies_per_question", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def canonical_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    if value is None:
        return None
    return str(value).strip()


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return json.dumps(canonical_value(value), ensure_ascii=False)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def build_select_variant(sql: str, select_exprs: Sequence[str]) -> str:
    sql0 = (sql or "").strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    return f"SELECT {', '.join(select_exprs)} {sql0[match.start():]};"


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.execute((sql or "").strip().rstrip(";"))
    cols = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall() if cur.description else []
    return cols, rows


def sql_references_column(sql: str, column_name: str) -> bool:
    sql_text = (sql or "").strip()
    if not sql_text or not column_name:
        return False
    quoted = quote_ident(column_name)
    if quoted in sql_text:
        return True
    return bool(re.search(rf"\b{re.escape(column_name)}\b", sql_text, flags=re.IGNORECASE))


def rows_to_objects(cols: Sequence[str], rows: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        obj = {}
        for idx, col in enumerate(cols):
            obj[col] = canonical_value(row[idx] if idx < len(row) else None)
        out.append(obj)
    return out


def row_key(row_obj: Dict[str, Any]) -> str:
    return json.dumps({k: canonical_value(v) for k, v in row_obj.items()}, ensure_ascii=False, sort_keys=True)


def dedupe_preserve(values: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for value in values:
        key = json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical_value(value))
    return out


def build_actual_mapping(
    visible_cols: Sequence[str],
    rows_with_hidden: Sequence[Tuple[Any, ...]],
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    visible_count = len(visible_cols)
    for row in rows_with_hidden:
        visible_obj = {
            visible_cols[idx]: canonical_value(row[idx] if idx < len(row) else None)
            for idx in range(visible_count)
        }
        hidden_value = canonical_value(row[visible_count] if len(row) > visible_count else None)
        key = row_key(visible_obj)
        entry = mapping.setdefault(
            key,
            {
                "visible_row": visible_obj,
                "actual_values": [],
            },
        )
        entry["actual_values"].append(hidden_value)
    for entry in mapping.values():
        entry["actual_values"] = dedupe_preserve(entry["actual_values"])
    return mapping


def preferred_evidence_columns(schema_cols: Sequence[str], selected_cols: Sequence[str]) -> List[str]:
    preferred = [col for col in ("NCT", "Trial name") if col in set(schema_cols)]
    out: List[str] = []
    seen = set()
    for col in preferred + list(selected_cols):
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def try_enrich_result_sql(
    *,
    base_sql: str,
    schema_cols: Sequence[str],
    selected_cols: Sequence[str],
    conn: sqlite3.Connection,
) -> Tuple[str, List[str], List[Tuple[Any, ...]], str]:
    enriched_cols = preferred_evidence_columns(schema_cols, selected_cols)
    if not enriched_cols:
        return base_sql, list(selected_cols), [], ""
    enriched_sql = build_select_variant(base_sql, [quote_ident(col) for col in enriched_cols])
    try:
        cols, rows = execute_query(conn, enriched_sql)
        return enriched_sql, cols, rows, ""
    except Exception as exc:
        return enriched_sql, list(selected_cols), [], str(exc)


def build_ctgov_metadata_bundle(
    *,
    row_objects: Sequence[Dict[str, Any]],
    timeout: float,
    max_studies: int,
    logger,
) -> Dict[str, Any]:
    ncts_in_order: List[str] = []
    for row in row_objects:
        nct = str(row.get("NCT") or "").strip()
        if nct and nct not in ncts_in_order:
            ncts_in_order.append(nct)
    if max_studies > 0:
        ncts_in_order = ncts_in_order[: max_studies]

    summaries_by_nct: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for nct in ncts_in_order:
        try:
            summaries_by_nct[nct] = summarize_study(fetch_study_by_nct(nct, timeout=timeout))
        except Exception as exc:
            errors[nct] = str(exc)
            logger.warning("ClinicalTrials.gov fetch failed for %s: %s", nct, str(exc).splitlines()[0] if str(exc) else "")

    rows_with_metadata: List[Dict[str, Any]] = []
    for idx, row in enumerate(row_objects, start=1):
        nct = str(row.get("NCT") or "").strip()
        rows_with_metadata.append(
            {
                "row_index": idx,
                "nct": nct,
                "trial_name": row.get("Trial name"),
                "ctgov_summary": summaries_by_nct.get(nct),
                "ctgov_error": errors.get(nct, ""),
            }
        )

    return {
        "ncts": ncts_in_order,
        "summaries_by_nct": summaries_by_nct,
        "errors_by_nct": errors,
        "rows_with_metadata": rows_with_metadata,
    }


def result_signature(cols: Sequence[str], rows: Sequence[Tuple[Any, ...]]) -> Tuple[Tuple[str, ...], Tuple[Tuple[Any, ...], ...]]:
    normalized_rows = [tuple(canonical_value(v) for v in row) for row in rows]
    sortable = sorted(
        (
            json.dumps(list(row), ensure_ascii=False, sort_keys=True),
            row,
        )
        for row in normalized_rows
    )
    return tuple(cols), tuple(row for _, row in sortable)


def result_sets_exact_match(
    cols_a: Sequence[str],
    rows_a: Sequence[Tuple[Any, ...]],
    cols_b: Sequence[str],
    rows_b: Sequence[Tuple[Any, ...]],
) -> bool:
    return result_signature(cols_a, rows_a) == result_signature(cols_b, rows_b)


def compute_similarity(predicted: Any, actual_options: Sequence[Any]) -> float:
    pred = canonical_value(predicted)
    actuals = [canonical_value(x) for x in actual_options]
    if pred in actuals:
        return 1.0
    pred_text = normalize_text(pred)
    if not pred_text or not actuals:
        return 0.0
    return max((1.0 if pred_text == normalize_text(x) else 0.0) for x in actuals)


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return (sum(values) / len(values)) if values else 0.0


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def filter_rows_by_csv_row_number(rows: Sequence[Dict[str, str]], csv_row_number: int) -> List[Dict[str, str]]:
    if not csv_row_number:
        return list(rows)
    if csv_row_number < 2:
        raise ValueError("csv_row_number must be 2 or greater because row 1 is the header")
    idx = csv_row_number - 2
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"csv_row_number {csv_row_number} is out of bounds")
    return [rows[idx]]


def _chat_completions_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def post_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        _chat_completions_url(api_base),
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=float(timeout)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_chat_completion_text(resp_obj: Dict[str, Any]) -> str:
    choices = resp_obj.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
        return "".join(parts).strip()
    return str(content or "").strip()


def completion_meta(resp_obj: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp_obj.get("choices") or []
    choice0 = choices[0] if choices else {}
    usage = resp_obj.get("usage") or {}
    return {
        "response_id": resp_obj.get("id"),
        "finish_reason": (choice0 or {}).get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def run_one_call_with_retries(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    num_retries: int,
    logger,
    stage: str,
) -> Dict[str, Any]:
    attempts = max(1, int(num_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            return post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            retryable = is_retryable_provider_error(exc)
            if attempt >= attempts or not retryable:
                logger.warning(
                    "%s call failed (attempt %d/%d): %s",
                    stage,
                    attempt,
                    attempts,
                    str(exc).splitlines()[0] if str(exc) else "",
                )
                raise
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retrying %s call attempt=%d/%d after %.1fs (err=%s)",
                stage,
                attempt,
                attempts,
                backoff,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            time.sleep(backoff)


def parse_inference_payload(raw_text: str) -> List[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return []
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    data: Any
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []

    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            data = data["predictions"]
        elif isinstance(data.get("rows"), list):
            data = data["rows"]
        elif isinstance(data.get("items"), list):
            data = data["items"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        row_index = item.get("row_index")
        if row_index is None:
            for key in ("index", "row", "row_id"):
                if key in item:
                    row_index = item[key]
                    break
        try:
            row_index_int = int(row_index) if row_index is not None else None
        except Exception:
            row_index_int = None

        answer = item.get("answer")
        if answer is None:
            for key in ("predicted_value", "value", "prediction", "inferred_value"):
                if key in item:
                    answer = item[key]
                    break

        out.append(
            {
                "row_index": row_index_int,
                "answer": canonical_value(answer),
            }
        )
    return out


def build_sql_prompt(
    *,
    question: str,
    table_name: str,
    visible_columns: Sequence[str],
    schema_hints_text: str,
    prompt_source_column_mode: str,
    source_column: str,
) -> str:
    schema_inline = ", ".join(f'"{col}"' for col in visible_columns)
    has_trial_name = "Trial name" in set(visible_columns)
    has_nct = "NCT" in set(visible_columns)
    lines = [
        f'You are writing one SQLite SELECT query over "{table_name}".',
        "Use the query to retrieve visible evidence needed to answer the question.",
        "The schema shown below is the only available schema.",
        "Do not compute the final transformed, classified, normalized, or inferred answer in SQL.",
        "Do not use CASE or hard-coded label mappings to synthesize the final answer.",
        "Prefer returning the minimal visible evidence rows needed for downstream reasoning.",
        "When available, include trial-level identifiers that help disambiguate studies.",
        "Rules:",
        "- Output ONLY the SQL query.",
        "- The SQL MUST start with SELECT.",
        f"- Use only columns from this visible schema: {schema_inline}",
        "- Put double quotes around EVERY column name exactly as shown in the visible schema.",
        "- Use single quotes for string literals.",
        "- Do not reference any unavailable or invented column.",
    ]
    if prompt_source_column_mode == "shown":
        lines.append(f'- If useful, prefer retrieving evidence from "{source_column}".')
    if has_trial_name and has_nct:
        lines.append('- If useful, include both "Trial name" and "NCT" in the SELECT output to identify the study.')
    elif has_trial_name:
        lines.append('- If useful, include "Trial name" in the SELECT output to identify the study.')
    elif has_nct:
        lines.append('- If useful, include "NCT" in the SELECT output to identify the study.')
    if schema_hints_text:
        lines.extend(["", schema_hints_text])
    lines.extend(["", f"Question: {question}", "SQL:"])
    return "\n".join(lines).strip()


def build_inference_prompt(
    *,
    question: str,
    target_column: str,
    result_rows: Sequence[Dict[str, Any]],
    prompt_source_column_mode: str,
    source_column: str,
    ctgov_rows: Optional[Sequence[Dict[str, Any]]] = None,
    ctgov_focus: str = "",
) -> str:
    has_trial_name = any("Trial name" in row for row in result_rows)
    has_nct = any("NCT" in row for row in result_rows)
    lines = [
        "You are answering a clinical_trials question using rows returned by a SQLite query.",
        "Infer the target field value for each returned row from the visible evidence in that row.",
        "When the row includes a trial identifier, you may use your internal knowledge about that specific study.",
        "If focused ClinicalTrials.gov study metadata is provided, use it when it helps answer the question.",
        "Return ONLY valid JSON in this exact shape:",
        '{"predictions":[{"row_index": 1, "answer": "..."}]}',
        "Use one prediction for each row shown below.",
        "Use JSON booleans, numbers, strings, or lists when appropriate.",
        "Do not add explanations.",
        "",
        f"Question: {question}",
        f"Target field to infer: {target_column}",
        "Use the question to understand which studies were retrieved, but make the final prediction match the target field.",
    ]
    if prompt_source_column_mode == "shown":
        lines.append(f'If useful, focus on evidence associated with "{source_column}".')
    if has_trial_name and has_nct:
        lines.append('Use "Trial name" and "NCT" to identify the exact study and rely on your knowledge of that trial when helpful.')
    elif has_trial_name:
        lines.append('Use "Trial name" to identify the study and rely on your knowledge of that trial when helpful.')
    elif has_nct:
        lines.append('Use "NCT" to identify the study and rely on your knowledge of that trial when helpful.')
    if ctgov_focus:
        lines.append(f"ClinicalTrials.gov evidence focus for this question: {ctgov_focus}")
    if target_column == "Control regimen":
        lines.extend(
            [
                "",
                "Few-shot examples for Control regimen normalization:",
                'Example 1',
                'Row: {"NCT": "NCT02788279", "Trial name": "IMblaze370", "Treatment regimen": "Atezolizumab+Cobimetinib"}',
                'Focused CTGov evidence: {"selected_ctgov_evidence": {"candidate_control_arms": [{"label": "Regorafenib", "intervention_names": ["Drug: Regorafenib"]}]}}',
                'Output: {"predictions":[{"row_index": 1, "answer": "Regorafenib"}]}',
                "",
                'Example 2',
                'Row: {"NCT": "NCT01984242", "Trial name": "IMmotion150", "Treatment regimen": "Atezolizumab+Bevacizumab"}',
                'Focused CTGov evidence: {"selected_ctgov_evidence": {"candidate_control_arms": [{"label": "Sunitinib", "intervention_names": ["Drug: Sunitinib"]}]}}',
                'Output: {"predictions":[{"row_index": 1, "answer": "Sunitinib"}]}',
                "",
                'Example 3',
                'Row: {"NCT": "NCT02763579", "Trial name": "IMpower133", "Treatment regimen": "Atezolizumab+Carboplatin+Etoposide"}',
                'Focused CTGov evidence: {"selected_ctgov_evidence": {"candidate_control_arms": [{"label": "Placebo + Carboplatin + Etoposide", "intervention_names": ["Drug: Carboplatin", "Drug: Etoposide", "Drug: Placebo"]}]}}',
                'Output: {"predictions":[{"row_index": 1, "answer": "Placebo+Carboplatin+Etoposide"}]}',
                "",
                'Example 4',
                'Row: {"NCT": "NCT02366143", "Trial name": "IMPower150", "Treatment regimen": "Atezolizumab+Carboplatin+Paclitaxel"}',
                'Focused CTGov evidence: {"selected_ctgov_evidence": {"candidate_control_arms": [{"label": "Arm C (Bevacizumab+Paclitaxel+Carboplatin)"}]}}',
                'Output: {"predictions":[{"row_index": 1, "answer": "Bevacizumab+Carboplatin+Paclitaxel"}]}',
                "",
                'Example 5',
                'Row: {"NCT": "NCT02425891", "Trial name": "IMpassion130", "Treatment regimen": "Atezolizumab+Nab-paclitaxel"}',
                'Focused CTGov evidence: {"selected_ctgov_evidence": {"candidate_control_arms": [{"label": "Placebo Plus Nab-Paclitaxel", "intervention_names": ["Drug: Nab-Paclitaxel", "Drug: Placebo"]}]}}',
                'Output: {"predictions":[{"row_index": 1, "answer": "Nab-paclitaxel"}]}',
                "",
                "Normalization hints:",
                '- Prefer comparator or control-arm evidence over the experimental treatment regimen.',
                '- Use candidate_control_arms first when available.',
                '- Remove arm labels like "Arm C (...)" and return only the normalized regimen.',
                '- Strip prefixes like "Drug:" and "Other:".',
                '- Join multi-agent regimens with "+".',
            ]
        )
    lines.append("Returned rows:")
    for idx, row_obj in enumerate(result_rows, start=1):
        lines.append(f"{idx}. {json.dumps(row_obj, ensure_ascii=False, sort_keys=True)}")
    ctgov_rows = list(ctgov_rows or [])
    if ctgov_rows:
        lines.append("")
        lines.append("Focused ClinicalTrials.gov study metadata:")
        for entry in ctgov_rows:
            row_index = entry.get("row_index")
            payload: Dict[str, Any] = {
                "row_index": row_index,
                "nct": entry.get("nct"),
                "trial_name": entry.get("trial_name"),
            }
            selected = entry.get("selected_ctgov_evidence")
            if isinstance(selected, dict):
                payload["selected_ctgov_evidence"] = selected
            else:
                summary = entry.get("ctgov_summary")
                if isinstance(summary, dict):
                    payload["brief_title"] = summary.get("brief_title")
                    payload["acronym"] = summary.get("acronym")
                    payload["candidate_control_arms"] = summary.get("candidate_control_arms")
                    payload["interventions"] = summary.get("interventions")
            if entry.get("ctgov_error"):
                payload["ctgov_error"] = entry.get("ctgov_error")
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines).strip()


def _compact_ctgov_nested_item(item: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    if field_name == "candidate_control_arms":
        compact: Dict[str, Any] = {}
        for key in ("label", "type", "intervention_names"):
            if item.get(key) is not None:
                compact[key] = item.get(key)
        return compact
    if field_name == "arm_groups":
        compact = {}
        for key in ("label", "type", "intervention_names"):
            if item.get(key) is not None:
                compact[key] = item.get(key)
        return compact
    if field_name == "interventions":
        compact = {}
        for key in ("name", "type", "arm_group_labels"):
            if item.get(key) is not None:
                compact[key] = item.get(key)
        return compact
    return dict(item)


def compact_ctgov_rows_for_prompt(
    ctgov_rows: Sequence[Dict[str, Any]],
    *,
    max_items_per_field: int = 3,
) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for entry in ctgov_rows:
        payload: Dict[str, Any] = {
            "row_index": entry.get("row_index"),
            "nct": entry.get("nct"),
            "trial_name": entry.get("trial_name"),
        }
        selected = entry.get("selected_ctgov_evidence")
        if isinstance(selected, dict):
            compact_selected: Dict[str, Any] = {}
            for field_name, value in selected.items():
                if isinstance(value, list):
                    compact_selected[field_name] = [
                        _compact_ctgov_nested_item(item, field_name)
                        for item in value[:max_items_per_field]
                        if isinstance(item, dict)
                    ]
                else:
                    compact_selected[field_name] = value
            payload["selected_ctgov_evidence"] = compact_selected
        if entry.get("ctgov_error"):
            payload["ctgov_error"] = entry.get("ctgov_error")
        compact_rows.append(payload)
    return compact_rows


def build_questions(
    rows: Sequence[Dict[str, str]],
    *,
    schema_cols: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    schema_set = set(schema_cols)
    items: List[Dict[str, Any]] = []
    skipped = Counter()

    for csv_row_number, row in enumerate(rows, start=2):
        question = (row.get(args.question_key) or "").strip()
        gt_sql = (row.get(args.gt_sql_key) or "").strip()
        source_column = (row.get(args.column_used_key) or "").strip()
        hidden_column = (row.get(args.final_column_key) or "").strip()

        if not hidden_column or hidden_column == "no_match" or hidden_column not in schema_set:
            skipped["final_column_not_usable"] += 1
            continue
        if not source_column or source_column not in schema_set:
            skipped["column_used_not_usable"] += 1
            continue
        if bool(args.require_distinct_source_target) and source_column == hidden_column:
            skipped["source_equals_hidden"] += 1
            continue
        if not question:
            skipped["missing_question"] += 1
            continue
        if not gt_sql:
            skipped["missing_gt_sql"] += 1
            continue

        items.append(
            {
                "item_id": f"row_{csv_row_number}",
                "csv_row_number": csv_row_number,
                "question": question,
                "gt_sql": gt_sql,
                "column_used": source_column,
                "final_column": hidden_column,
            }
        )
        if args.limit and len(items) >= args.limit:
            break

    return items, dict(skipped)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip())
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"methodv2_sql_infer_{run_dir.name}")

    logger.info("CSV path: %s", csv_path)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("API base: %s", args.api_base)
    logger.info("Model name: %s", args.model_name)
    logger.info("Prompt source-column mode: %s", args.prompt_source_column_mode)
    logger.info("Dry run: %s", bool(args.dry_run))
    checkpoint_enabled = not bool(args.dry_run)

    question_checkpoint_jsonl = run_dir / "question_results_checkpoint.jsonl"
    row_checkpoint_jsonl = run_dir / "row_level_checkpoint.jsonl"
    sql_requests_jsonl = run_dir / "sql_requests.jsonl"
    sql_responses_jsonl = run_dir / "sql_responses.jsonl"
    planner_requests_jsonl = run_dir / "planner_requests.jsonl"
    planner_responses_jsonl = run_dir / "planner_responses.jsonl"
    infer_requests_jsonl = run_dir / "inference_requests.jsonl"
    infer_responses_jsonl = run_dir / "inference_responses.jsonl"

    rows = filter_rows_by_csv_row_number(load_csv_rows(csv_path), args.csv_row_number)
    conn = sqlite3.connect(db_path)
    try:
        schema_cols = fetch_schema(conn, args.table_name)
    finally:
        conn.close()

    items, skipped = build_questions(rows, schema_cols=schema_cols, args=args)
    question_rows: List[Dict[str, Any]] = load_jsonl_rows(question_checkpoint_jsonl) if checkpoint_enabled else []
    row_level_rows: List[Dict[str, Any]] = load_jsonl_rows(row_checkpoint_jsonl) if checkpoint_enabled else []
    sql_request_rows: List[Dict[str, Any]] = load_jsonl_rows(sql_requests_jsonl) if checkpoint_enabled else []
    sql_response_rows: List[Dict[str, Any]] = load_jsonl_rows(sql_responses_jsonl) if checkpoint_enabled else []
    planner_request_rows: List[Dict[str, Any]] = load_jsonl_rows(planner_requests_jsonl) if checkpoint_enabled else []
    planner_response_rows: List[Dict[str, Any]] = load_jsonl_rows(planner_responses_jsonl) if checkpoint_enabled else []
    infer_request_rows: List[Dict[str, Any]] = load_jsonl_rows(infer_requests_jsonl) if checkpoint_enabled else []
    infer_response_rows: List[Dict[str, Any]] = load_jsonl_rows(infer_responses_jsonl) if checkpoint_enabled else []

    completed_item_ids = {str(row.get("item_id")) for row in question_rows if row.get("item_id")}
    row_level_seen = set()
    for row in row_level_rows:
        if row.get("item_id") is None or row.get("row_index") is None:
            continue
        try:
            row_level_seen.add((str(row.get("item_id")), int(row.get("row_index"))))
        except Exception:
            continue
    sql_request_seen = {str(row.get("item_id")) for row in sql_request_rows if row.get("item_id")}
    sql_response_seen = {str(row.get("item_id")) for row in sql_response_rows if row.get("item_id")}
    planner_request_seen = {str(row.get("item_id")) for row in planner_request_rows if row.get("item_id")}
    planner_response_seen = {str(row.get("item_id")) for row in planner_response_rows if row.get("item_id")}
    infer_request_seen = {str(row.get("item_id")) for row in infer_request_rows if row.get("item_id")}
    infer_response_seen = {str(row.get("item_id")) for row in infer_response_rows if row.get("item_id")}

    if checkpoint_enabled and completed_item_ids:
        logger.info("Loaded %d completed question checkpoints from %s", len(completed_item_ids), question_checkpoint_jsonl)
        items = [item for item in items if item["item_id"] not in completed_item_ids]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["final_column"]].append(item)

    logger.info("Eligible questions: %d", len(items))
    logger.info("Skipped: %s", skipped)
    logger.info("Groups: %d", len(grouped))
    for hidden_column, group in sorted(grouped.items()):
        logger.info("  %s -> %d rows", hidden_column, len(group))

    meta = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "api_base": args.api_base,
        "model_name": args.model_name,
        "prompt_source_column_mode": args.prompt_source_column_mode,
        "csv_row_number": args.csv_row_number,
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
        "use_ctgov_metadata": bool(args.use_ctgov_metadata),
        "use_ctgov_hybrid_planner": bool(args.use_ctgov_hybrid_planner),
        "loaded_completed_item_ids": len(completed_item_ids),
        "eligible_questions": len(items),
        "skipped": skipped,
        "groups": {col: len(group) for col, group in sorted(grouped.items())},
    }
    write_json(run_dir / "run_meta.json", meta)
    questions_root_dir = run_dir / "questions"

    original_conn = sqlite3.connect(db_path)
    try:
        for hidden_column, group in sorted(grouped.items()):
            group_dir = run_dir / f"group__{sanitize_name(hidden_column)}"
            group_dir.mkdir(parents=True, exist_ok=True)

            reduced_db_path = group_dir / "reduced.db"
            create_reduced_db(
                source_db=db_path,
                output_db=reduced_db_path,
                table_name=args.table_name,
                removed_column=hidden_column,
            )
            reduced_conn = sqlite3.connect(reduced_db_path)
            try:
                visible_schema = fetch_schema(reduced_conn, args.table_name)
                schema_hints = fetch_schema_value_hints(reduced_conn, args.table_name, visible_schema)
                schema_hints_text = render_schema_hints(visible_schema, schema_hints)

                for item in group:
                    gt_visible_cols: List[str] = []
                    gt_visible_rows: List[Tuple[Any, ...]] = []
                    gt_visible_error = ""
                    gt_visible_source_db = "reduced"
                    gt_evidence_sql = ""
                    gt_evidence_cols: List[str] = []
                    gt_evidence_rows: List[Tuple[Any, ...]] = []
                    gt_evidence_error = ""
                    gt_actual_sql = ""
                    gt_actual_mapping: Dict[str, Dict[str, Any]] = {}
                    gt_actual_error = ""

                    gt_sql_uses_hidden_column = sql_references_column(item["gt_sql"], hidden_column)
                    gt_query_conn = original_conn if gt_sql_uses_hidden_column else reduced_conn
                    gt_query_schema = list(schema_cols) if gt_sql_uses_hidden_column else list(visible_schema)
                    gt_visible_source_db = "original" if gt_sql_uses_hidden_column else "reduced"

                    try:
                        gt_visible_cols, gt_visible_rows = execute_query(gt_query_conn, item["gt_sql"])
                    except Exception as exc:
                        gt_visible_error = str(exc)
                        if not gt_sql_uses_hidden_column and sql_references_column(gt_visible_error, hidden_column):
                            try:
                                gt_visible_cols, gt_visible_rows = execute_query(original_conn, item["gt_sql"])
                                gt_visible_error = ""
                                gt_query_conn = original_conn
                                gt_query_schema = list(schema_cols)
                                gt_visible_source_db = "original_fallback"
                            except Exception as fallback_exc:
                                gt_visible_error = str(fallback_exc)

                    if gt_visible_cols:
                        try:
                            gt_evidence_sql, gt_evidence_cols, gt_evidence_rows, gt_evidence_error = try_enrich_result_sql(
                                base_sql=item["gt_sql"],
                                schema_cols=gt_query_schema,
                                selected_cols=gt_visible_cols,
                                conn=gt_query_conn,
                            )
                            if gt_evidence_error or not gt_evidence_rows:
                                gt_evidence_cols, gt_evidence_rows = gt_visible_cols, gt_visible_rows
                                gt_evidence_sql = item["gt_sql"]
                            gt_actual_sql = build_select_variant(
                                gt_evidence_sql or item["gt_sql"],
                                [quote_ident(col) for col in gt_evidence_cols] + [quote_ident(hidden_column)],
                            )
                            _, gt_actual_rows = execute_query(original_conn, gt_actual_sql)
                            gt_actual_mapping = build_actual_mapping(gt_evidence_cols, gt_actual_rows)
                        except Exception as exc:
                            gt_actual_error = str(exc)

                    sql_prompt = build_sql_prompt(
                        question=item["question"],
                        table_name=args.table_name,
                        visible_columns=visible_schema,
                        schema_hints_text=schema_hints_text,
                        prompt_source_column_mode=args.prompt_source_column_mode,
                        source_column=item["column_used"],
                    )
                    sql_request_row = {
                        "stage": "sql_generation",
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": item["question"],
                        "column_used": item["column_used"],
                        "final_column": hidden_column,
                        "prompt_source_column_mode": args.prompt_source_column_mode,
                        "prompt": sql_prompt,
                    }
                    if item["item_id"] not in sql_request_seen:
                        sql_request_rows.append(sql_request_row)
                        if checkpoint_enabled:
                            append_jsonl(str(sql_requests_jsonl), sql_request_row)
                        sql_request_seen.add(item["item_id"])

                    sql_raw_output = ""
                    pred_sql = ""
                    sql_call_error = ""
                    sql_meta: Dict[str, Any] = {}
                    pred_visible_cols: List[str] = []
                    pred_visible_rows: List[Tuple[Any, ...]] = []
                    pred_visible_error = ""
                    sql_result_exact_match = 0

                    pred_evidence_sql = ""
                    pred_evidence_cols: List[str] = []
                    pred_evidence_rows: List[Tuple[Any, ...]] = []
                    pred_evidence_error = ""
                    pred_actual_sql = ""
                    pred_actual_mapping: Dict[str, Dict[str, Any]] = {}
                    pred_actual_error = ""
                    ctgov_bundle: Dict[str, Any] = {}
                    ctgov_plan: Dict[str, Any] = {}
                    ctgov_selected_rows: List[Dict[str, Any]] = []
                    ctgov_planner_prompt = ""
                    ctgov_planner_raw_output = ""
                    ctgov_planner_error = ""
                    ctgov_planner_meta: Dict[str, Any] = {}
                    ctgov_metadata_path = ""
                    ctgov_prompt_rows: List[Dict[str, Any]] = []

                    infer_prompt = ""
                    infer_raw_output = ""
                    infer_call_error = ""
                    infer_meta: Dict[str, Any] = {}
                    parsed_predictions: List[Dict[str, Any]] = []

                    prompted_row_objects: List[Dict[str, Any]] = []
                    matched_rows = 0
                    total_rows = 0
                    similarity_sum = 0.0

                    if not args.dry_run:
                        try:
                            sql_resp = run_one_call_with_retries(
                                api_base=args.api_base,
                                api_key=args.api_key,
                                model_name=args.model_name,
                                prompt=sql_prompt,
                                temperature=args.sql_temperature,
                                top_p=args.sql_top_p,
                                max_tokens=args.sql_max_tokens,
                                timeout=args.timeout,
                                num_retries=args.num_retries,
                                logger=logger,
                                stage="sql_generation",
                            )
                            sql_raw_output = parse_chat_completion_text(sql_resp)
                            pred_sql = extract_sql_candidate(sql_raw_output)
                            sql_meta = completion_meta(sql_resp)
                        except Exception as exc:
                            sql_call_error = str(exc)

                    if pred_sql:
                        try:
                            pred_visible_cols, pred_visible_rows = execute_query(reduced_conn, pred_sql)
                            sql_result_exact_match = int(
                                result_sets_exact_match(pred_visible_cols, pred_visible_rows, gt_visible_cols, gt_visible_rows)
                            )
                        except Exception as exc:
                            pred_visible_error = str(exc)

                    pred_visible_row_objects = rows_to_objects(pred_visible_cols, pred_visible_rows)

                    if pred_sql and pred_visible_cols:
                        try:
                            pred_evidence_sql, pred_evidence_cols, pred_evidence_rows, pred_evidence_error = try_enrich_result_sql(
                                base_sql=pred_sql,
                                schema_cols=visible_schema,
                                selected_cols=pred_visible_cols,
                                conn=reduced_conn,
                            )
                            if pred_evidence_error or not pred_evidence_rows:
                                pred_evidence_cols, pred_evidence_rows = pred_visible_cols, pred_visible_rows
                                pred_evidence_sql = pred_sql
                            pred_actual_sql = build_select_variant(
                                pred_evidence_sql or pred_sql,
                                [quote_ident(col) for col in pred_evidence_cols] + [quote_ident(hidden_column)],
                            )
                            _, pred_actual_rows = execute_query(original_conn, pred_actual_sql)
                            pred_actual_mapping = build_actual_mapping(pred_evidence_cols, pred_actual_rows)
                        except Exception as exc:
                            pred_actual_error = str(exc)

                    pred_evidence_row_objects = rows_to_objects(pred_evidence_cols, pred_evidence_rows)
                    if args.max_prompt_rows and len(pred_evidence_row_objects) > args.max_prompt_rows:
                        prompted_row_objects = pred_evidence_row_objects[: args.max_prompt_rows]
                    else:
                        prompted_row_objects = list(pred_evidence_row_objects)

                    if args.use_ctgov_metadata and pred_evidence_row_objects and not args.dry_run:
                        ctgov_bundle = build_ctgov_metadata_bundle(
                            row_objects=pred_evidence_row_objects,
                            timeout=args.ctgov_timeout,
                            max_studies=args.ctgov_max_studies_per_question,
                            logger=logger,
                        )
                        if bool(args.use_ctgov_hybrid_planner):
                            ctgov_planner_prompt = build_ctgov_planning_prompt(
                                question=item["question"],
                                target_column=hidden_column,
                                result_rows=prompted_row_objects,
                                prompt_source_column_mode=args.prompt_source_column_mode,
                                source_column=item["column_used"],
                                ctgov_bundle=ctgov_bundle,
                            )
                            planner_request_row = {
                                "stage": "ctgov_planning",
                                "item_id": item["item_id"],
                                "csv_row_number": item["csv_row_number"],
                                "question": item["question"],
                                "column_used": item["column_used"],
                                "final_column": hidden_column,
                                "prompt_source_column_mode": args.prompt_source_column_mode,
                                "prompt": ctgov_planner_prompt,
                            }
                            if item["item_id"] not in planner_request_seen:
                                planner_request_rows.append(planner_request_row)
                                if checkpoint_enabled:
                                    append_jsonl(str(planner_requests_jsonl), planner_request_row)
                                planner_request_seen.add(item["item_id"])
                            try:
                                planner_resp = run_one_call_with_retries(
                                    api_base=args.api_base,
                                    api_key=args.api_key,
                                    model_name=args.model_name,
                                    prompt=ctgov_planner_prompt,
                                    temperature=args.planner_temperature,
                                    top_p=args.planner_top_p,
                                    max_tokens=args.planner_max_tokens,
                                    timeout=args.timeout,
                                    num_retries=args.num_retries,
                                    logger=logger,
                                    stage="ctgov_planning",
                                )
                                ctgov_planner_raw_output = parse_chat_completion_text(planner_resp)
                                ctgov_planner_meta = completion_meta(planner_resp)
                                ctgov_plan = parse_ctgov_plan(ctgov_planner_raw_output)
                            except Exception as exc:
                                ctgov_planner_error = str(exc)
                                ctgov_plan = default_ctgov_plan("planner call failed")
                            planner_response_row = {
                                "stage": "ctgov_planning",
                                "item_id": item["item_id"],
                                "csv_row_number": item["csv_row_number"],
                                "question": item["question"],
                                "column_used": item["column_used"],
                                "final_column": hidden_column,
                                "prompt_source_column_mode": args.prompt_source_column_mode,
                                "raw_output": ctgov_planner_raw_output,
                                "parsed_plan_json": json.dumps(ctgov_plan, ensure_ascii=False),
                                "error": ctgov_planner_error,
                                "model_meta_json": json.dumps(ctgov_planner_meta, ensure_ascii=False),
                            }
                            if item["item_id"] not in planner_response_seen:
                                planner_response_rows.append(planner_response_row)
                                if checkpoint_enabled:
                                    append_jsonl(str(planner_responses_jsonl), planner_response_row)
                                planner_response_seen.add(item["item_id"])
                        if not ctgov_plan:
                            ctgov_plan = default_ctgov_plan("fallback default plan")
                    if prompted_row_objects and ctgov_bundle and bool(ctgov_plan.get("need_ctgov_metadata", True)):
                        ctgov_selected_rows = select_ctgov_evidence_for_prompt(
                            result_rows=prompted_row_objects,
                            ctgov_bundle=ctgov_bundle,
                            plan=ctgov_plan,
                        )
                        ctgov_prompt_rows = compact_ctgov_rows_for_prompt(ctgov_selected_rows)

                    if prompted_row_objects and not args.dry_run:
                        infer_prompt = build_inference_prompt(
                            question=item["question"],
                            target_column=hidden_column,
                            result_rows=prompted_row_objects,
                            prompt_source_column_mode=args.prompt_source_column_mode,
                            source_column=item["column_used"],
                            ctgov_rows=ctgov_prompt_rows,
                            ctgov_focus=str(ctgov_plan.get("focus") or ""),
                        )
                        infer_request_row = {
                            "stage": "value_inference",
                            "item_id": item["item_id"],
                            "csv_row_number": item["csv_row_number"],
                            "question": item["question"],
                            "column_used": item["column_used"],
                            "final_column": hidden_column,
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "pred_sql": pred_sql,
                            "prompt": infer_prompt,
                            "visible_rows_json": json.dumps(prompted_row_objects, ensure_ascii=False),
                            "ctgov_plan_json": json.dumps(ctgov_plan, ensure_ascii=False),
                            "ctgov_rows_json": json.dumps(ctgov_prompt_rows, ensure_ascii=False),
                        }
                        if item["item_id"] not in infer_request_seen:
                            infer_request_rows.append(infer_request_row)
                            if checkpoint_enabled:
                                append_jsonl(str(infer_requests_jsonl), infer_request_row)
                            infer_request_seen.add(item["item_id"])
                        try:
                            infer_resp = run_one_call_with_retries(
                                api_base=args.api_base,
                                api_key=args.api_key,
                                model_name=args.model_name,
                                prompt=infer_prompt,
                                temperature=args.infer_temperature,
                                top_p=args.infer_top_p,
                                max_tokens=args.infer_max_tokens,
                                timeout=args.timeout,
                                num_retries=args.num_retries,
                                logger=logger,
                                stage="value_inference",
                            )
                            infer_raw_output = parse_chat_completion_text(infer_resp)
                            infer_meta = completion_meta(infer_resp)
                            parsed_predictions = parse_inference_payload(infer_raw_output)
                        except Exception as exc:
                            infer_call_error = str(exc)

                    pred_by_row_index = {
                        int(pred.get("row_index")): canonical_value(pred.get("answer"))
                        for pred in parsed_predictions
                        if pred.get("row_index") is not None
                    }

                    question_export_paths = export_question_tables(
                        questions_root_dir=questions_root_dir,
                        item_id=item["item_id"],
                        question=item["question"],
                        hidden_column=hidden_column,
                        group_name=sanitize_name(hidden_column),
                        prompt_mode=args.prompt_source_column_mode,
                        gt_sql=gt_evidence_sql or item["gt_sql"],
                        pred_sql=pred_evidence_sql or pred_sql,
                        gt_visible_cols=gt_evidence_cols or gt_visible_cols,
                        gt_visible_rows=rows_to_objects(gt_evidence_cols or gt_visible_cols, gt_evidence_rows or gt_visible_rows),
                        pred_visible_cols=pred_evidence_cols or pred_visible_cols,
                        pred_visible_rows=pred_evidence_row_objects or pred_visible_row_objects,
                        pred_by_row_index=pred_by_row_index,
                        actual_mapping=gt_actual_mapping,
                        row_key_fn=row_key,
                    )
                    if ctgov_bundle:
                        ctgov_metadata_path = str(Path(question_export_paths["question_dir"]) / "ctgov_metadata.json")
                        write_json(
                            ctgov_metadata_path,
                            {
                                "bundle": ctgov_bundle,
                                "plan": ctgov_plan,
                                "planner_prompt": ctgov_planner_prompt,
                                "planner_raw_output": ctgov_planner_raw_output,
                                "planner_error": ctgov_planner_error,
                                "planner_meta": ctgov_planner_meta,
                                "selected_rows": ctgov_selected_rows,
                                "prompt_rows": ctgov_prompt_rows,
                            },
                        )

                    for row_index, row_obj in enumerate(prompted_row_objects, start=1):
                        actual_entry = pred_actual_mapping.get(
                            row_key(row_obj),
                            {"visible_row": row_obj, "actual_values": []},
                        )
                        actual_values = actual_entry.get("actual_values") or []
                        predicted_value = pred_by_row_index.get(row_index)
                        similarity = compute_similarity(predicted_value, actual_values)
                        matched_rows += int(similarity >= 1.0 - 1e-12)
                        similarity_sum += similarity
                        total_rows += 1
                        row_level_row = {
                            "item_id": item["item_id"],
                            "csv_row_number": item["csv_row_number"],
                            "question": item["question"],
                            "column_used": item["column_used"],
                            "final_column": hidden_column,
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "row_index": row_index,
                            "visible_row_json": json.dumps(row_obj, ensure_ascii=False, sort_keys=True),
                            "pred_sql": pred_evidence_sql or pred_sql,
                            "predicted_hidden_value": format_value(predicted_value),
                            "actual_hidden_values_json": json.dumps(actual_values, ensure_ascii=False),
                            "exact_match": int(similarity >= 1.0 - 1e-12),
                            "similarity": similarity,
                            "sql_result_exact_match": sql_result_exact_match,
                        }
                        row_key_tuple = (item["item_id"], row_index)
                        if row_key_tuple not in row_level_seen:
                            row_level_rows.append(row_level_row)
                            if checkpoint_enabled:
                                append_jsonl(str(row_checkpoint_jsonl), row_level_row)
                            row_level_seen.add(row_key_tuple)

                    exact_match_rate = (matched_rows / total_rows) if total_rows else 0.0
                    avg_similarity = (similarity_sum / total_rows) if total_rows else 0.0

                    sql_response_row = {
                        "stage": "sql_generation",
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": item["question"],
                        "column_used": item["column_used"],
                        "final_column": hidden_column,
                        "prompt_source_column_mode": args.prompt_source_column_mode,
                        "raw_output": sql_raw_output,
                        "pred_sql": pred_sql,
                        "error": sql_call_error or pred_visible_error,
                        "model_meta_json": json.dumps(sql_meta, ensure_ascii=False),
                    }
                    if item["item_id"] not in sql_response_seen:
                        sql_response_rows.append(sql_response_row)
                        if checkpoint_enabled:
                            append_jsonl(str(sql_responses_jsonl), sql_response_row)
                        sql_response_seen.add(item["item_id"])

                    if infer_prompt or infer_raw_output or infer_call_error:
                        infer_response_row = {
                            "stage": "value_inference",
                            "item_id": item["item_id"],
                            "csv_row_number": item["csv_row_number"],
                            "question": item["question"],
                            "column_used": item["column_used"],
                            "final_column": hidden_column,
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "pred_sql": pred_sql,
                            "raw_output": infer_raw_output,
                            "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                            "error": infer_call_error,
                            "model_meta_json": json.dumps(infer_meta, ensure_ascii=False),
                        }
                        if item["item_id"] not in infer_response_seen:
                            infer_response_rows.append(infer_response_row)
                            if checkpoint_enabled:
                                append_jsonl(str(infer_responses_jsonl), infer_response_row)
                            infer_response_seen.add(item["item_id"])

                    question_row = {
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": item["question"],
                        "column_used": item["column_used"],
                        "final_column": hidden_column,
                        "prompt_source_column_mode": args.prompt_source_column_mode,
                        "gt_sql": item["gt_sql"],
                        "gt_visible_columns_json": json.dumps(gt_visible_cols, ensure_ascii=False),
                        "gt_visible_rows_json": json.dumps(rows_to_objects(gt_visible_cols, gt_visible_rows), ensure_ascii=False),
                        "gt_visible_row_count": len(gt_visible_rows),
                        "gt_visible_source_db": gt_visible_source_db,
                        "gt_visible_exec_error": gt_visible_error,
                        "gt_evidence_sql": gt_evidence_sql,
                        "gt_evidence_columns_json": json.dumps(gt_evidence_cols, ensure_ascii=False),
                        "gt_evidence_rows_json": json.dumps(rows_to_objects(gt_evidence_cols, gt_evidence_rows), ensure_ascii=False),
                        "gt_evidence_row_count": len(gt_evidence_rows),
                        "gt_evidence_exec_error": gt_evidence_error,
                        "gt_actual_sql": gt_actual_sql,
                        "gt_actual_mapping_json": json.dumps(gt_actual_mapping, ensure_ascii=False),
                        "gt_actual_exec_error": gt_actual_error,
                        "sql_prompt": sql_prompt,
                        "sql_raw_output": sql_raw_output,
                        "pred_sql": pred_sql,
                        "pred_evidence_sql": pred_evidence_sql,
                        "sql_call_error": sql_call_error,
                        "pred_visible_exec_error": pred_visible_error,
                        "pred_visible_columns_json": json.dumps(pred_visible_cols, ensure_ascii=False),
                        "pred_visible_rows_json": json.dumps(pred_visible_row_objects, ensure_ascii=False),
                        "pred_visible_row_count": len(pred_visible_rows),
                        "pred_evidence_exec_error": pred_evidence_error,
                        "pred_evidence_columns_json": json.dumps(pred_evidence_cols, ensure_ascii=False),
                        "pred_evidence_rows_json": json.dumps(pred_evidence_row_objects, ensure_ascii=False),
                        "pred_evidence_row_count": len(pred_evidence_rows),
                        "sql_result_exact_match": sql_result_exact_match,
                        "pred_actual_sql": pred_actual_sql,
                        "pred_actual_mapping_json": json.dumps(pred_actual_mapping, ensure_ascii=False),
                        "pred_actual_exec_error": pred_actual_error,
                        "ctgov_metadata_json": json.dumps(ctgov_bundle, ensure_ascii=False),
                        "ctgov_metadata_path": ctgov_metadata_path,
                        "ctgov_plan_json": json.dumps(ctgov_plan, ensure_ascii=False),
                        "ctgov_selected_rows_json": json.dumps(ctgov_selected_rows, ensure_ascii=False),
                        "ctgov_prompt_rows_json": json.dumps(ctgov_prompt_rows, ensure_ascii=False),
                        "ctgov_planner_prompt": ctgov_planner_prompt,
                        "ctgov_planner_raw_output": ctgov_planner_raw_output,
                        "ctgov_planner_error": ctgov_planner_error,
                        "ctgov_planner_meta_json": json.dumps(ctgov_planner_meta, ensure_ascii=False),
                        "infer_prompt": infer_prompt,
                        "infer_raw_output": infer_raw_output,
                        "infer_call_error": infer_call_error,
                        "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                        "prompted_row_count": len(prompted_row_objects),
                        "matched_rows": matched_rows,
                        "total_rows": total_rows,
                        "exact_match_rate": exact_match_rate,
                        "avg_similarity": avg_similarity,
                        "question_dir": question_export_paths["question_dir"],
                        "final_table_csv": question_export_paths["final_table_csv"],
                        "ground_truth_table_csv": question_export_paths["ground_truth_table_csv"],
                    }
                    if item["item_id"] not in completed_item_ids:
                        question_rows.append(question_row)
                        if checkpoint_enabled:
                            append_jsonl(str(question_checkpoint_jsonl), question_row)
                        completed_item_ids.add(item["item_id"])

                write_jsonl(group_dir / "sql_requests.jsonl", [r for r in sql_request_rows if r["final_column"] == hidden_column])
                write_jsonl(group_dir / "sql_responses.jsonl", [r for r in sql_response_rows if r["final_column"] == hidden_column])
                write_jsonl(group_dir / "planner_requests.jsonl", [r for r in planner_request_rows if r["final_column"] == hidden_column])
                write_jsonl(group_dir / "planner_responses.jsonl", [r for r in planner_response_rows if r["final_column"] == hidden_column])
                write_jsonl(group_dir / "inference_requests.jsonl", [r for r in infer_request_rows if r["final_column"] == hidden_column])
                write_jsonl(group_dir / "inference_responses.jsonl", [r for r in infer_response_rows if r["final_column"] == hidden_column])
            finally:
                reduced_conn.close()
    finally:
        original_conn.close()

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

    question_csv = run_dir / "all_question_results.csv"
    row_csv = run_dir / "row_level_predictions.csv"
    sql_requests_jsonl = run_dir / "sql_requests.jsonl"
    sql_responses_jsonl = run_dir / "sql_responses.jsonl"
    planner_requests_jsonl = run_dir / "planner_requests.jsonl"
    planner_responses_jsonl = run_dir / "planner_responses.jsonl"
    infer_requests_jsonl = run_dir / "inference_requests.jsonl"
    infer_responses_jsonl = run_dir / "inference_responses.jsonl"

    write_csv(question_csv, question_rows)
    write_csv(row_csv, row_level_rows)
    if checkpoint_enabled:
        write_jsonl(question_checkpoint_jsonl, question_rows)
        write_jsonl(row_checkpoint_jsonl, row_level_rows)
    write_jsonl(sql_requests_jsonl, sql_request_rows)
    write_jsonl(sql_responses_jsonl, sql_response_rows)
    write_jsonl(planner_requests_jsonl, planner_request_rows)
    write_jsonl(planner_responses_jsonl, planner_response_rows)
    write_jsonl(infer_requests_jsonl, infer_request_rows)
    write_jsonl(infer_responses_jsonl, infer_response_rows)

    overall = {
        "meta": meta,
        "question_count": len(question_rows),
        "row_prediction_count": len(row_level_rows),
        "avg_sql_result_exact_match": avg_numeric(question_rows, "sql_result_exact_match"),
        "avg_inference_exact_match_rate": avg_numeric(question_rows, "exact_match_rate"),
        "avg_inference_similarity": avg_numeric(question_rows, "avg_similarity"),
        "outputs": {
            "all_question_results_csv": str(question_csv),
            "row_level_predictions_csv": str(row_csv),
            "questions_dir": str(questions_root_dir),
            "question_checkpoint_jsonl": str(question_checkpoint_jsonl) if checkpoint_enabled else "",
            "row_level_checkpoint_jsonl": str(row_checkpoint_jsonl) if checkpoint_enabled else "",
            "sql_requests_jsonl": str(sql_requests_jsonl),
            "sql_responses_jsonl": str(sql_responses_jsonl),
            "planner_requests_jsonl": str(planner_requests_jsonl),
            "planner_responses_jsonl": str(planner_responses_jsonl),
            "inference_requests_jsonl": str(infer_requests_jsonl),
            "inference_responses_jsonl": str(infer_responses_jsonl),
        },
    }
    write_json(run_dir / "summary.json", overall)
    logger.info("Wrote all-question results CSV: %s", question_csv)
    logger.info("Wrote row-level predictions CSV: %s", row_csv)
    logger.info("Wrote summary JSON: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
