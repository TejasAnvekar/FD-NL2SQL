#!/usr/bin/env python3
"""
Friend-style runner, but via vLLM OpenAI-compatible SERVER API.

Adds:
- Pydantic schema output (JSON only) enforced with vLLM guided decoding (guided_json)
- structured_logprobs support (structured_logprobs.add_logprobs if installed)
  + fallback to OpenAI-compatible token logprobs
- Batched inference via concurrency; vLLM server continuous-batches automatically

Docs:
- Structured outputs / guided_json in vLLM OpenAI server
- Continuous batching via concurrent requests
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel

try:
    from structured_logprobs import add_logprobs  # optional
except Exception:
    add_logprobs = None


# -------------------------
# Pydantic response schemas
# -------------------------

class SQLOnlyResponse(BaseModel):
    sql: str


class COTResponse(BaseModel):
    thinking: str
    sql: str


# -------------------------
# Args
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SQL-R1-7B on a question slice via vLLM OpenAI server.")

    # vLLM OpenAI server
    ap.add_argument("--api_base", required=True, help="e.g. http://localhost:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", required=True, help="vLLM served model name")

    # data
    ap.add_argument("--input_json", default="data/natural_question_1500.json")
    ap.add_argument("--schema_json", default="data/schema.json")
    ap.add_argument("--output_json", default="results/nl-sql-r1-vllm.json")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10, help="Use -1 for all remaining rows")
    ap.add_argument("--question_keys", default="natural_question,question,original_question,new_question")
    ap.add_argument("--table_name", default="clinical_trials")

    # generation
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=120.0)

    # batching (client concurrency; server does continuous batching)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--batch_concurrency", type=int, default=8)

    # retry
    ap.add_argument("--num_retries", type=int, default=2)

    # structured output + logprobs
    ap.add_argument("--prompt_style", choices=["sql_only", "cot"], default="sql_only")
    ap.add_argument("--use_pydantic_schema", type=int, default=1)
    ap.add_argument("--logprob_mode", choices=["structured", "token", "none"], default="structured")
    ap.add_argument("--prompt_logprobs", type=int, default=0, help="vLLM extra param (prompt token logprobs)")
    ap.add_argument("--top_logprobs", type=int, default=0, help="If supported; otherwise keep 0")

    return ap.parse_args()


# -------------------------
# Small helpers (same as friend code)
# -------------------------

def parse_comma_keys(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def pick_first_nonempty(row: Dict[str, Any], keys: List[str]) -> Tuple[str, Optional[str]]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            return sv, k
    return "", None


def load_schema_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("schema_json must be a JSON list of column names")
    cols = [str(x).strip() for x in obj if str(x).strip()]
    if not cols:
        raise ValueError("No columns found in schema_json")
    return cols


def build_messages(table_name: str, schema_cols: List[str], question: str, prompt_style: str) -> List[Dict[str, str]]:
    col_list = ", ".join([f'"{c}"' for c in schema_cols])

    if prompt_style == "cot":
        system = (
            "You are an expert NL2SQL assistant. "
            "Return JSON only (no markdown), with keys 'thinking' and 'sql'."
        )
        user = f"""Table name:
{table_name}

Available columns:
{col_list}

Rules:
- Use only table "{table_name}".
- Use only listed columns.
- Keep all filters and numeric thresholds from the question.
- Output JSON only, like: {{"thinking":"...","sql":"SELECT ...;"}}

Question:
{question}
"""
    else:
        system = (
            "You are an expert NL2SQL assistant. "
            "Return JSON only (no markdown), with key 'sql'."
        )
        user = f"""Table name:
{table_name}

Available columns:
{col_list}

Rules:
- Use only table "{table_name}".
- Use only listed columns.
- Keep all filters and numeric thresholds from the question.
- Output JSON only, like: {{"sql":"SELECT ...;"}}

Question:
{question}
"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_sql_fallback(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    m = re.search(r"\b(SELECT|WITH)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():].strip()
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t


def is_retryable_provider_error(exc: Exception) -> bool:
    s = str(exc or "").lower()
    markers = [
        "rate limit", "too many requests", "429",
        "503", "service unavailable", "temporarily unavailable",
        "deadline exceeded", "timed out",
        "connection reset", "connection error",
        "502", "504",
    ]
    return any(m in s for m in markers)


# -------------------------
# Logprob utilities
# -------------------------

def exp_structure(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: exp_structure(v) for k, v in x.items()}
    if isinstance(x, list):
        return [exp_structure(v) for v in x]
    if isinstance(x, (int, float)):
        try:
            return float(math.exp(x))
        except Exception:
            return None
    return x


def flatten_numbers(x: Any) -> List[float]:
    if isinstance(x, dict):
        out: List[float] = []
        for v in x.values():
            out.extend(flatten_numbers(v))
        return out
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            out.extend(flatten_numbers(v))
        return out
    if isinstance(x, (int, float)):
        return [float(x)]
    return []


def structured_logprob_payload(completion: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[float]]:
    """Uses structured_logprobs.add_logprobs if available."""
    if add_logprobs is None:
        return {}, {}, None
    wrapped = add_logprobs(completion)
    log_probs = getattr(wrapped, "log_probs", None)
    field_logprobs = log_probs[0] if isinstance(log_probs, list) and log_probs else {}
    field_confidence = exp_structure(field_logprobs)
    nums = flatten_numbers(field_logprobs)
    conf_overall = float(math.exp(sum(nums) / len(nums))) if nums else None
    return field_logprobs, field_confidence, conf_overall


def token_logprob_payload(completion: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[float]]:
    """Fallback for OpenAI-compatible token logprobs."""
    choices = getattr(completion, "choices", []) or []
    c0 = choices[0] if choices else None
    if c0 is None:
        return {}, {}, None

    c0_logprobs = getattr(c0, "logprobs", None)
    content = getattr(c0_logprobs, "content", None) if c0_logprobs is not None else None
    if not isinstance(content, list):
        return {}, {}, None

    toks: List[str] = []
    lps: List[Optional[float]] = []
    for tok in content:
        token_text = getattr(tok, "token", None)
        lp = getattr(tok, "logprob", None)
        toks.append(str(token_text) if token_text is not None else "")
        lps.append(float(lp) if isinstance(lp, (int, float)) else None)

    finite = [x for x in lps if isinstance(x, (int, float))]
    conf_overall = float(math.exp(sum(finite) / len(finite))) if finite else None

    field_logprobs = {"tokens": toks, "token_logprobs": lps, "token_count": len(toks)}
    field_confidence = {"token_confidence": exp_structure(lps)}
    return field_logprobs, field_confidence, conf_overall


# -------------------------
# vLLM OpenAI API call (with retry) + batching via concurrency
# -------------------------

def guided_schema_for(prompt_style: str) -> Dict[str, Any]:
    model_cls = COTResponse if prompt_style == "cot" else SQLOnlyResponse
    return model_cls.model_json_schema()


def call_one_with_retries(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    guided_schema: Optional[Dict[str, Any]],
) -> Any:
    attempts = max(1, int(args.num_retries) + 1)

    extra_body: Dict[str, Any] = {}
    if int(args.use_pydantic_schema) and guided_schema is not None:
        # vLLM guided decoding: guided_json :contentReference[oaicite:5]{index=5}
        extra_body["guided_json"] = guided_schema

    if args.logprob_mode in ("structured", "token"):
        # vLLM supports passing non-OpenAI params via extra_body :contentReference[oaicite:6]{index=6}
        extra_body["prompt_logprobs"] = int(args.prompt_logprobs)

    for attempt in range(1, attempts + 1):
        try:
            req: Dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_tokens": int(args.max_tokens),
                "timeout": float(args.timeout),
            }

            if int(args.use_pydantic_schema):
                # Encourage JSON response
                req["response_format"] = {"type": "json_object"}

            if args.logprob_mode in ("structured", "token"):
                req["logprobs"] = True
                if int(args.top_logprobs) > 0:
                    # If your openai-python doesn't accept this, just set --top_logprobs 0
                    req["top_logprobs"] = int(args.top_logprobs)

            if extra_body:
                req["extra_body"] = extra_body

            return client.chat.completions.create(**req)

        except Exception as e:
            if attempt >= attempts or not is_retryable_provider_error(e):
                raise
            backoff = min(30.0, 2.0 ** (attempt - 1))
            time.sleep(backoff)


def run_batch(
    client: OpenAI,
    args: argparse.Namespace,
    batch_messages: List[List[Dict[str, str]]],
    guided_schema: Optional[Dict[str, Any]],
) -> List[Union[Any, Exception]]:
    out: List[Union[Any, Exception]] = [Exception("uninitialized")] * len(batch_messages)
    workers = max(1, min(int(args.batch_concurrency), len(batch_messages)))

    # Concurrency here -> vLLM server continuous batching :contentReference[oaicite:7]{index=7}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(call_one_with_retries, client, args, msgs, guided_schema): i
            for i, msgs in enumerate(batch_messages)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                out[i] = e
    return out


# -------------------------
# Main (mirrors friend code)
# -------------------------

def main() -> None:
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("input_json must be a JSON list")

    start = max(0, int(args.start))
    picked = rows[start:]
    if int(args.limit) > -1:
        picked = picked[: int(args.limit)]

    schema_cols = load_schema_columns(args.schema_json)
    q_keys = parse_comma_keys(args.question_keys)

    guided_schema = guided_schema_for(args.prompt_style) if int(args.use_pydantic_schema) else None

    # Build entries (same as friend)
    entries: List[Dict[str, Any]] = []
    for off, row in enumerate(picked):
        gidx = start + off
        if not isinstance(row, dict):
            entries.append(
                {
                    "row_index": gidx,
                    "item_id": None,
                    "original_question": "",
                    "gt_sql": "",
                    "pred_sql": "",
                    "raw_text": "",
                    "status": "error",
                    "error": "ROW_NOT_OBJECT",
                }
            )
            continue

        q, qk = pick_first_nonempty(row, q_keys)
        if not q:
            entries.append(
                {
                    "row_index": gidx,
                    "item_id": row.get("item_id"),
                    "original_question": "",
                    "gt_sql": str(row.get("gt_sql") or ""),
                    "pred_sql": "",
                    "raw_text": "",
                    "status": "error",
                    "error": f"MISSING_QUESTION_KEYS:{args.question_keys}",
                }
            )
            continue

        entries.append(
            {
                "row_index": gidx,
                "item_id": row.get("item_id"),
                "question_key": qk,
                "original_question": q,
                "gt_sql": str(row.get("gt_sql") or ""),
                "messages": build_messages(args.table_name, schema_cols, q, args.prompt_style),
                "pred_sql": "",
                "raw_text": "",
                "status": "pending",
                "error": None,
                # logprob extras
                "field_logprobs": {},
                "field_confidence": {},
                "confidence_overall": None,
            }
        )

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    pending_idx = [i for i, e in enumerate(entries) if e.get("status") == "pending"]
    bs = max(1, int(args.batch_size))

    for b0 in range(0, len(pending_idx), bs):
        batch_ids = pending_idx[b0 : b0 + bs]
        batch_messages = [entries[i]["messages"] for i in batch_ids]

        outs = run_batch(client, args, batch_messages, guided_schema)

        for i, out in zip(batch_ids, outs):
            if isinstance(out, Exception):
                entries[i]["status"] = "error"
                entries[i]["error"] = str(out).splitlines()[0] if str(out) else "PROVIDER_ERROR"
                entries[i].pop("messages", None)
                continue

            # raw output
            raw_text = ""
            try:
                raw_text = (out.choices[0].message.content or "").strip()
            except Exception:
                raw_text = ""
            entries[i]["raw_text"] = raw_text

            # parse JSON -> sql
            pred_sql = ""
            if int(args.use_pydantic_schema):
                try:
                    obj = json.loads(raw_text)
                    pred_sql = str(obj.get("sql") or "").strip()
                except Exception:
                    pred_sql = extract_sql_fallback(raw_text)
            else:
                pred_sql = extract_sql_fallback(raw_text)

            entries[i]["pred_sql"] = pred_sql
            entries[i]["status"] = "ok" if pred_sql else "empty"
            entries[i]["error"] = None if pred_sql else "EMPTY_SQL"

            # logprobs
            if args.logprob_mode == "structured":
                fld, fconf, conf = structured_logprob_payload(out)
                if not fld:
                    fld, fconf, conf = token_logprob_payload(out)
                entries[i]["field_logprobs"] = fld or {}
                entries[i]["field_confidence"] = fconf or {}
                entries[i]["confidence_overall"] = conf
            elif args.logprob_mode == "token":
                fld, fconf, conf = token_logprob_payload(out)
                entries[i]["field_logprobs"] = fld or {}
                entries[i]["field_confidence"] = fconf or {}
                entries[i]["confidence_overall"] = conf

            entries[i].pop("messages", None)

        done = min(b0 + len(batch_ids), len(pending_idx))
        print(f"Processed {done}/{len(pending_idx)}")

    for e in entries:
        e.pop("messages", None)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    ok = sum(1 for e in entries if e.get("status") == "ok")
    empty = sum(1 for e in entries if e.get("status") == "empty")
    err = sum(1 for e in entries if e.get("status") == "error")
    print("\n=== SQL-R1 vLLM API RUN ===")
    print(f"Input slice rows: {len(entries)}")
    print(f"OK:               {ok}")
    print(f"EMPTY:            {empty}")
    print(f"ERROR:            {err}")
    print(f"Output JSON:      {args.output_json}")


if __name__ == "__main__":
    main()