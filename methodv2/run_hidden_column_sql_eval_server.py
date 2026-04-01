#!/usr/bin/env python3
"""Hidden-column SQL experiment runner using an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib import error as urllib_error
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
    aggregate_eval_summaries,
    build_hidden_column_prompt,
    collect_items,
    create_reduced_db,
    extract_sql_candidate,
    fetch_schema,
    load_csv_rows,
    make_run_dir,
    render_schema_hints,
    run_evaluator,
)
from utils import (
    fetch_schema_value_hints,
    is_retryable_provider_error,
    setup_logger,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run hidden-column SQL generation experiments against a local "
            "OpenAI-compatible vLLM server and evaluate them with eval_run_baselines_v2.py."
        )
    )
    ap.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--question_key", default="natural_language_query")
    ap.add_argument("--gt_sql_key", default="sql_query")
    ap.add_argument("--final_column_key", default="final_column")
    ap.add_argument("--expected_response_key", default="expected_llm_response")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="", help="Optional fixed run directory name.")
    ap.add_argument("--limit", type=int, default=0, help="0 means all eligible rows.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batch_concurrency", type=int, default=4)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="gemma-3-4b-it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument("--max_eval_rows", type=int, default=10000)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare reduced DBs, prompts, and GT files but skip model generation and evaluation.",
    )
    return ap.parse_args()


def chunked(seq: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int]]:
    size = max(1, int(batch_size))
    start = 0
    while start < len(seq):
        end = min(len(seq), start + size)
        yield start, end
        start = end


def _chat_completions_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _post_chat_completion(
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib_request.Request(
        _chat_completions_url(api_base),
        data=payload,
        headers=headers,
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
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content or "").strip()


def completion_meta(resp_obj: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp_obj.get("choices") or []
    choice0 = choices[0] if choices else {}
    usage = resp_obj.get("usage") or {}
    return {
        "provider_call_ok": True,
        "response_id": resp_obj.get("id"),
        "finish_reason": (choice0 or {}).get("finish_reason"),
        "stop_reason": None,
        "token_ids_len": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _run_one_call_with_retries(
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
            return _post_chat_completion(
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
                    "OpenAI-compatible call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    str(exc).splitlines()[0] if str(exc) else "",
                )
                raise
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retrying OpenAI-compatible call attempt=%d/%d after %.1fs (err=%s)",
                attempt,
                attempts,
                backoff,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            time.sleep(backoff)


def run_batch(
    *,
    prompts: Sequence[str],
    args: argparse.Namespace,
    logger,
) -> List[Any]:
    outputs: List[Any] = [None] * len(prompts)
    max_workers = max(1, min(int(args.batch_concurrency), len(prompts) if prompts else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_call_with_retries,
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                num_retries=args.num_retries,
                logger=logger,
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                outputs[idx] = future.result()
            except Exception as exc:
                outputs[idx] = exc
    return outputs


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip())
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"methodv2_server_{run_dir.name}")

    logger.info("CSV path: %s", csv_path)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("API base: %s", args.api_base)
    logger.info("Model name: %s", args.model_name)
    logger.info("Dry run: %s", bool(args.dry_run))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    raw_rows = load_csv_rows(csv_path)
    conn = sqlite3.connect(db_path)
    try:
        original_schema = fetch_schema(conn, args.table_name)
    finally:
        conn.close()

    items, skipped = collect_items(
        raw_rows,
        schema_cols=original_schema,
        question_key=args.question_key,
        gt_sql_key=args.gt_sql_key,
        final_column_key=args.final_column_key,
        expected_response_key=args.expected_response_key,
        limit=args.limit,
    )

    grouped_items: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        grouped_items[item.removed_column].append(item)

    logger.info("Eligible rows: %d", len(items))
    logger.info("Skipped rows: %s", skipped)
    logger.info("Groups: %d", len(grouped_items))
    for column_name, group in sorted(grouped_items.items()):
        logger.info("  %s -> %d rows", column_name, len(group))

    meta = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "table_name": args.table_name,
        "api_base": args.api_base,
        "model_name": args.model_name,
        "dry_run": bool(args.dry_run),
        "eligible_items": len(items),
        "skipped": skipped,
        "groups": {col: len(group) for col, group in sorted(grouped_items.items())},
    }
    write_json(run_dir / "run_meta.json", meta)

    all_gt_rows = [
        {
            "item_id": item.item_id,
            args.question_key: item.question,
            args.gt_sql_key: item.gt_sql,
            "removed_column": item.removed_column,
            "csv_row_number": item.csv_row_number,
            "expected_llm_response": item.expected_llm_response,
        }
        for item in items
    ]
    write_jsonl(run_dir / "all_gt.jsonl", all_gt_rows)

    if not items:
        logger.warning("No eligible items found. Nothing to run.")
        return

    group_results: List[Dict[str, Any]] = []
    all_pred_rows: List[Dict[str, Any]] = []

    for removed_column, group in sorted(grouped_items.items()):
        group_dir = run_dir / f"group__{removed_column.replace('/', '_').replace(' ', '_')}"
        group_dir.mkdir(parents=True, exist_ok=True)

        reduced_db_path = group_dir / "reduced.db"
        visible_schema = create_reduced_db(
            source_db=db_path,
            output_db=reduced_db_path,
            table_name=args.table_name,
            removed_column=removed_column,
        )
        logger.info("Prepared reduced DB for group=%s at %s", removed_column, reduced_db_path)

        reduced_conn = sqlite3.connect(reduced_db_path)
        try:
            schema_hints = fetch_schema_value_hints(reduced_conn, args.table_name, visible_schema)
        finally:
            reduced_conn.close()
        schema_hints_text = render_schema_hints(visible_schema, schema_hints)

        gt_rows = []
        prompt_rows = []
        prompts: List[str] = []
        for item in group:
            prompt = build_hidden_column_prompt(
                question=item.question,
                table_name=args.table_name,
                visible_columns=visible_schema,
                schema_hints_text=schema_hints_text,
            )
            gt_rows.append(
                {
                    "item_id": item.item_id,
                    args.question_key: item.question,
                    args.gt_sql_key: item.gt_sql,
                    "removed_column": removed_column,
                    "csv_row_number": item.csv_row_number,
                    "expected_llm_response": item.expected_llm_response,
                }
            )
            prompt_rows.append(
                {
                    "item_id": item.item_id,
                    "removed_column": removed_column,
                    "question": item.question,
                    "prompt": prompt,
                }
            )
            prompts.append(prompt)

        gt_path = group_dir / "gt.jsonl"
        prompts_path = group_dir / "prompts.jsonl"
        pred_path = group_dir / "pred.jsonl"
        eval_path = group_dir / "eval.json"

        write_jsonl(gt_path, gt_rows)
        write_jsonl(prompts_path, prompt_rows)

        group_result: Dict[str, Any] = {
            "removed_column": removed_column,
            "group_dir": str(group_dir),
            "reduced_db_path": str(reduced_db_path),
            "items": len(group),
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "eval_path": str(eval_path),
            "dry_run": bool(args.dry_run),
        }

        if args.dry_run:
            logger.info("Dry run enabled; skipping API calls/eval for group=%s", removed_column)
            group_results.append(group_result)
            continue

        pred_rows: List[Dict[str, Any]] = []
        for start, end in chunked(prompts, args.batch_size):
            batch_prompts = prompts[start:end]
            batch_items = group[start:end]
            outputs = run_batch(prompts=batch_prompts, args=args, logger=logger)

            for item, out_obj in zip(batch_items, outputs):
                raw_text = ""
                pred_sql = ""
                error = None
                model_meta: Dict[str, Any] = {}

                try:
                    if isinstance(out_obj, Exception):
                        raise out_obj

                    raw_text = parse_chat_completion_text(out_obj)
                    pred_sql = extract_sql_candidate(raw_text)
                    model_meta = completion_meta(out_obj)
                except Exception as exc:
                    error = str(exc)

                pred_row = {
                    "item_id": item.item_id,
                    "question_used": item.question,
                    "removed_column": removed_column,
                    "csv_row_number": item.csv_row_number,
                    "pred_sql": pred_sql,
                    "raw_text": raw_text,
                    "error": error,
                    "api_base": args.api_base,
                    "model_name": args.model_name,
                    "model_meta": model_meta,
                }
                pred_rows.append(pred_row)
                all_pred_rows.append(pred_row)

        write_jsonl(pred_path, pred_rows)
        logger.info("Generated %d predictions for group=%s", len(pred_rows), removed_column)

        run_evaluator(
            pred_path=pred_path,
            gt_path=gt_path,
            db_path=reduced_db_path,
            output_json=eval_path,
            question_key=args.question_key,
            gt_sql_key=args.gt_sql_key,
            max_rows=args.max_eval_rows,
        )
        with eval_path.open("r", encoding="utf-8") as handle:
            eval_obj = json.load(handle)
        group_result["summary"] = eval_obj.get("summary", {})
        group_results.append(group_result)
        logger.info("Evaluated group=%s summary=%s", removed_column, group_result["summary"])

    if all_pred_rows:
        write_jsonl(run_dir / "all_pred.jsonl", all_pred_rows)

    summary = {
        "meta": meta,
        "group_results": group_results,
        "aggregate_summary": aggregate_eval_summaries(group_results) if not args.dry_run else None,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote summary to %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
