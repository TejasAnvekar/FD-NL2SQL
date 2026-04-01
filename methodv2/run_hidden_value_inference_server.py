#!/usr/bin/env python3
"""Infer hidden column values from visible source-column values via a local vLLM server."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urllib_request

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_hidden_column_sql_eval import (
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_RUN_ROOT,
    create_reduced_db,
    fetch_schema,
    load_csv_rows,
    make_run_dir,
    sanitize_name,
)
from utils import is_retryable_provider_error, quote_ident, setup_logger, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "For each eligible question, remove the hidden final_column, retrieve the visible "
            "column_used values, ask a local vLLM server to infer the missing final_column values, "
            "and write comparison CSVs."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--gt_sql_key", default="sql_query")
    ap.add_argument("--column_used_key", default="column_used")
    ap.add_argument("--final_column_key", default="final_column")
    ap.add_argument("--expected_response_key", default="expected_llm_response")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--require_distinct_source_target", type=int, default=1)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="gemma-3-4b-it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument("--batch_concurrency", type=int, default=4)
    ap.add_argument("--max_visible_values", type=int, default=50)
    ap.add_argument(
        "--prompt_source_column_mode",
        choices=("shown", "hidden"),
        default="hidden",
        help="Whether the prompt explicitly names the source column that produced the visible values.",
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


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


def json_from_expected(payload_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads((payload_text or "").strip())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def build_select_variant(sql: str, select_exprs: Sequence[str]) -> str:
    sql0 = (sql or "").strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    select_prefix = ", ".join(select_exprs)
    return f"SELECT {select_prefix} {sql0[match.start():]};"


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.execute((sql or "").strip().rstrip(";"))
    cols = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall() if cur.description else []
    return cols, rows


def unique_in_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for value in values:
        key = json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical_value(value))
    return out


def build_actual_mapping(rows: Sequence[Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for visible_value, hidden_value in rows:
        visible_key = format_value(visible_value)
        entry = mapping.setdefault(
            visible_key,
            {
                "visible_value": canonical_value(visible_value),
                "actual_values": [],
            },
        )
        entry["actual_values"].append(canonical_value(hidden_value))

    for entry in mapping.values():
        entry["actual_values"] = unique_in_order(entry["actual_values"])
    return mapping


def build_visible_list(rows: Sequence[Tuple[Any]]) -> List[Any]:
    return unique_in_order(row[0] for row in rows)


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
                    "Value-inference call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    str(exc).splitlines()[0] if str(exc) else "",
                )
                raise
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retrying value-inference call attempt=%d/%d after %.1fs (err=%s)",
                attempt,
                attempts,
                backoff,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            time.sleep(backoff)


def run_batch(requests: Sequence[Dict[str, Any]], args: argparse.Namespace, logger) -> List[Any]:
    outputs: List[Any] = [None] * len(requests)
    max_workers = max(1, min(int(args.batch_concurrency), len(requests) if requests else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_call_with_retries,
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                prompt=req["prompt"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                num_retries=args.num_retries,
                logger=logger,
            ): idx
            for idx, req in enumerate(requests)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                outputs[idx] = future.result()
            except Exception as exc:
                outputs[idx] = exc
    return outputs


def parse_prediction_payload(raw_text: str) -> List[Dict[str, Any]]:
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
        elif isinstance(data.get("items"), list):
            data = data["items"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        visible_value = item.get("visible_value")
        predicted_value = item.get("predicted_value")
        if predicted_value is None:
            for key in ("answer", "hidden_value", "value", "prediction", "predicted_hidden_value", "inferred_value"):
                if key in item:
                    predicted_value = item[key]
                    break
        if visible_value is None:
            for key in ("source_value", "input_value", "column_used_value"):
                if key in item:
                    visible_value = item[key]
                    break
        normalized.append(
            {
                "visible_value": canonical_value(visible_value),
                "predicted_value": canonical_value(predicted_value),
            }
        )
    return normalized


def build_inference_prompt(
    *,
    question: str,
    source_column: str,
    visible_values: Sequence[Any],
    prompt_source_column_mode: str,
) -> str:
    visible_lines = []
    for idx, value in enumerate(visible_values, start=1):
        visible_lines.append(f"{idx}. {format_value(value)}")

    lines = [
        "You are answering a clinical_trials question from values returned by a SQL query.",
        "The query was run against the currently available table schema.",
    ]
    if prompt_source_column_mode == "shown":
        lines.append(f'The returned values come from the column "{source_column}".')
    else:
        lines.append("The returned values come from the query result.")
    lines.extend(
        [
            "For each visible value below, answer the question as directly as possible.",
            "Return ONLY valid JSON in this exact shape:",
            '{"predictions":[{"visible_value": "...", "answer": "..."}]}',
            "Use JSON booleans/numbers/lists when appropriate.",
            "Preserve one answer per visible value shown below.",
            "Do not add explanations.",
            "",
            f"Question: {question}",
        ]
    )
    if prompt_source_column_mode == "shown":
        lines.append(f"Column returned by the SQL query: {source_column}")
    lines.extend(
        [
            "Visible values:",
            "\n".join(visible_lines),
        ]
    )
    return "\n".join(lines).strip()


def build_questions(
    rows: Sequence[Dict[str, str]],
    *,
    schema_cols: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    usable_schema = set(schema_cols)
    items: List[Dict[str, Any]] = []
    skipped = Counter()

    for csv_row_number, row in enumerate(rows, start=2):
        hidden_column = (row.get(args.final_column_key) or "").strip()
        source_column = (row.get(args.column_used_key) or "").strip()
        question = (row.get(args.question_key) or "").strip()
        gt_sql = (row.get(args.gt_sql_key) or "").strip()

        if not hidden_column or hidden_column == "no_match" or hidden_column not in usable_schema:
            skipped["final_column_not_usable"] += 1
            continue
        if not source_column or source_column not in usable_schema:
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
                "expected_llm_response": (row.get(args.expected_response_key) or "").strip(),
            }
        )
        if args.limit and len(items) >= args.limit:
            break

    return items, dict(skipped)


def compute_similarity(predicted: Any, actual_options: Sequence[Any]) -> float:
    pred = canonical_value(predicted)
    actuals = [canonical_value(x) for x in actual_options]
    if pred in actuals:
        return 1.0
    pred_text = normalize_text(pred)
    if not pred_text:
        return 0.0
    return max((1.0 if pred_text == normalize_text(x) else 0.0) for x in actuals) if actuals else 0.0


def avg_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return (sum(values) / len(values)) if values else 0.0


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip())
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"methodv2_value_{run_dir.name}")

    logger.info("CSV path: %s", csv_path)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("API base: %s", args.api_base)
    logger.info("Model name: %s", args.model_name)
    logger.info("Prompt source-column mode: %s", args.prompt_source_column_mode)
    logger.info("Dry run: %s", bool(args.dry_run))

    rows = load_csv_rows(csv_path)
    conn = sqlite3.connect(db_path)
    try:
        schema_cols = fetch_schema(conn, args.table_name)
    finally:
        conn.close()

    items, skipped = build_questions(rows, schema_cols=schema_cols, args=args)
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
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
        "eligible_questions": len(items),
        "skipped": skipped,
        "groups": {col: len(group) for col, group in sorted(grouped.items())},
    }
    write_json(run_dir / "run_meta.json", meta)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    request_rows: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []

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
                requests: List[Dict[str, Any]] = []
                prepared_items: List[Dict[str, Any]] = []

                for item in group:
                    source_column = item["column_used"]
                    visible_sql = item["gt_sql"]
                    actual_sql = build_select_variant(
                        item["gt_sql"],
                        [quote_ident(source_column), quote_ident(hidden_column)],
                    )

                    visible_cols, visible_rows = execute_query(reduced_conn, visible_sql)
                    _, actual_rows = execute_query(original_conn, actual_sql)

                    visible_values = build_visible_list(visible_rows)
                    if args.max_visible_values and len(visible_values) > args.max_visible_values:
                        visible_values = visible_values[: args.max_visible_values]

                    actual_mapping = build_actual_mapping(actual_rows)
                    actual_mapping_trimmed = {
                        key: value
                        for key, value in actual_mapping.items()
                        if canonical_value(value["visible_value"]) in visible_values
                    }

                    prompt = build_inference_prompt(
                        question=item["question"],
                        source_column=source_column,
                        visible_values=visible_values,
                        prompt_source_column_mode=args.prompt_source_column_mode,
                    )

                    prepared = dict(item)
                    prepared.update(
                        {
                            "visible_sql": visible_sql,
                            "actual_sql": actual_sql,
                            "visible_values": visible_values,
                            "actual_mapping": actual_mapping_trimmed,
                            "prompt": prompt,
                        }
                    )
                    prepared_items.append(prepared)
                    requests.append({"item_id": item["item_id"], "prompt": prompt})
                    request_rows.append(
                        {
                            "item_id": item["item_id"],
                            "csv_row_number": item["csv_row_number"],
                            "question": item["question"],
                            "column_used": source_column,
                            "final_column": hidden_column,
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "visible_sql": visible_sql,
                            "actual_sql": actual_sql,
                            "visible_values_json": json.dumps(visible_values, ensure_ascii=False),
                            "actual_mapping_json": json.dumps(actual_mapping_trimmed, ensure_ascii=False),
                            "prompt": prompt,
                        }
                    )

                write_jsonl(group_dir / "requests.jsonl", request_rows[-len(requests):] if requests else [])

                if args.dry_run:
                    for prepared in prepared_items:
                        summary_rows.append(
                            {
                                "item_id": prepared["item_id"],
                                "csv_row_number": prepared["csv_row_number"],
                                "question": prepared["question"],
                                "column_used": prepared["column_used"],
                                "final_column": prepared["final_column"],
                                "prompt_source_column_mode": args.prompt_source_column_mode,
                                "visible_value_count": len(prepared["visible_values"]),
                                "actual_value_count": len(prepared["actual_mapping"]),
                                "matched_items": 0,
                                "total_items": len(prepared["actual_mapping"]),
                                "exact_match_rate": 0.0,
                                "avg_similarity": 0.0,
                                "visible_sql": prepared["visible_sql"],
                                "actual_sql": prepared["actual_sql"],
                                "visible_values_json": json.dumps(prepared["visible_values"], ensure_ascii=False),
                                "llm_raw_output": "",
                                "parsed_predictions_json": "",
                                "actual_mapping_json": json.dumps(prepared["actual_mapping"], ensure_ascii=False),
                                "notes": "dry_run",
                            }
                        )
                    continue

                outputs = run_batch(requests, args=args, logger=logger)
                for prepared, out_obj in zip(prepared_items, outputs):
                    raw_text = ""
                    parsed_predictions: List[Dict[str, Any]] = []
                    error = ""
                    model_meta: Dict[str, Any] = {}

                    try:
                        if isinstance(out_obj, Exception):
                            raise out_obj
                        raw_text = parse_chat_completion_text(out_obj)
                        parsed_predictions = parse_prediction_payload(raw_text)
                        model_meta = completion_meta(out_obj)
                    except Exception as exc:
                        error = str(exc)

                    response_rows.append(
                        {
                            "item_id": prepared["item_id"],
                            "csv_row_number": prepared["csv_row_number"],
                            "question": prepared["question"],
                            "column_used": prepared["column_used"],
                            "final_column": prepared["final_column"],
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "visible_sql": prepared["visible_sql"],
                            "actual_sql": prepared["actual_sql"],
                            "visible_values_json": json.dumps(prepared["visible_values"], ensure_ascii=False),
                            "actual_mapping_json": json.dumps(prepared["actual_mapping"], ensure_ascii=False),
                            "prompt": prepared["prompt"],
                            "llm_raw_output": raw_text,
                            "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                            "error": error,
                            "model_meta_json": json.dumps(model_meta, ensure_ascii=False),
                        }
                    )

                    pred_map = {
                        format_value(pred.get("visible_value")): canonical_value(pred.get("predicted_value"))
                        for pred in parsed_predictions
                        if pred.get("visible_value") is not None
                    }
                    actual_mapping = prepared["actual_mapping"]
                    total_items = len(actual_mapping)
                    matched_items = 0
                    similarity_sum = 0.0

                    for visible_key, actual_entry in actual_mapping.items():
                        actual_values = actual_entry["actual_values"]
                        predicted_value = pred_map.get(visible_key)
                        similarity = compute_similarity(predicted_value, actual_values)
                        if similarity >= 1.0 - 1e-12:
                            matched_items += 1
                        similarity_sum += similarity
                        detail_rows.append(
                            {
                                "item_id": prepared["item_id"],
                                "csv_row_number": prepared["csv_row_number"],
                                "question": prepared["question"],
                                "column_used": prepared["column_used"],
                                "final_column": prepared["final_column"],
                                "prompt_source_column_mode": args.prompt_source_column_mode,
                                "visible_value": format_value(actual_entry["visible_value"]),
                                "actual_hidden_values_json": json.dumps(actual_values, ensure_ascii=False),
                                "predicted_hidden_value": format_value(predicted_value),
                                "exact_match": int(similarity >= 1.0 - 1e-12),
                                "similarity": similarity,
                            }
                        )

                    exact_match_rate = (matched_items / total_items) if total_items else 0.0
                    avg_similarity = (similarity_sum / total_items) if total_items else 0.0
                    summary_rows.append(
                        {
                            "item_id": prepared["item_id"],
                            "csv_row_number": prepared["csv_row_number"],
                            "question": prepared["question"],
                            "column_used": prepared["column_used"],
                            "final_column": prepared["final_column"],
                            "prompt_source_column_mode": args.prompt_source_column_mode,
                            "visible_value_count": len(prepared["visible_values"]),
                            "actual_value_count": len(actual_mapping),
                            "matched_items": matched_items,
                            "total_items": total_items,
                            "exact_match_rate": exact_match_rate,
                            "avg_similarity": avg_similarity,
                            "visible_values_json": json.dumps(prepared["visible_values"], ensure_ascii=False),
                            "actual_mapping_json": json.dumps(actual_mapping, ensure_ascii=False),
                            "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                            "llm_raw_output": raw_text,
                            "error": error,
                        }
                    )
            finally:
                reduced_conn.close()
    finally:
        original_conn.close()

    summary_csv = run_dir / "question_summary.csv"
    all_question_results_csv = run_dir / "all_question_results.csv"
    detail_csv = run_dir / "value_level_comparison.csv"
    requests_jsonl = run_dir / "requests.jsonl"
    responses_jsonl = run_dir / "responses.jsonl"

    def write_csv(path: Path, rows_out: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows_out:
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("")
            return
        fieldnames = list(rows_out[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

    write_csv(summary_csv, summary_rows)
    write_csv(all_question_results_csv, summary_rows)
    write_csv(detail_csv, detail_rows)
    write_jsonl(requests_jsonl, request_rows)
    write_jsonl(responses_jsonl, response_rows)

    overall = {
        "meta": meta,
        "question_count": len(summary_rows),
        "value_count": len(detail_rows),
        "avg_exact_match_rate": avg_numeric(summary_rows, "exact_match_rate"),
        "avg_similarity": avg_numeric(summary_rows, "avg_similarity"),
        "outputs": {
            "question_summary_csv": str(summary_csv),
            "all_question_results_csv": str(all_question_results_csv),
            "value_level_comparison_csv": str(detail_csv),
            "requests_jsonl": str(requests_jsonl),
            "responses_jsonl": str(responses_jsonl),
        },
    }
    write_json(run_dir / "summary.json", overall)
    logger.info("Wrote question summary CSV: %s", summary_csv)
    logger.info("Wrote all-question results CSV: %s", all_question_results_csv)
    logger.info("Wrote value-level comparison CSV: %s", detail_csv)
    logger.info("Wrote requests JSONL: %s", requests_jsonl)
    logger.info("Wrote summary JSON: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
