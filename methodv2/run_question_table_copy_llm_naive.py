#!/usr/bin/env python3
"""Naive full-table LLM baseline using only a copied table plus the question.

For each exported table-derivation question, this runner:

1. Loads the saved question and row-level ground truth.
2. Creates a per-question copy of the full `clinical_trials` table.
3. Hides any obvious answer-leaking column only when it is distinct from the
   source evidence column already used by the question.
4. Prompts a local OpenAI-compatible LLM with only:
   - the question
   - the copied visible table
5. Scores the row-wise JSON predictions against the saved ground-truth rows.

This is intentionally a naive baseline:
- no SQL generation
- no CTGov retrieval
- no hybrid planner
- no external evidence beyond the visible table copy
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urllib_request

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_embedding_cosine_baselines import avg_numeric, write_csv  # noqa: E402
from run_hidden_column_sql_eval import make_run_dir, sanitize_name  # noqa: E402
from utils import is_retryable_provider_error, setup_logger, write_json  # noqa: E402

DEFAULT_MANIFEST = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv"
DEFAULT_ANNOTATED_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv"
DEFAULT_DB_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a naive full-table LLM baseline that only sees a copied table "
            "and the question."
        )
    )
    ap.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    ap.add_argument("--annotated_csv", default=DEFAULT_ANNOTATED_CSV)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="question_table_copy_llm_naive_100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--csv_row_number", type=int, default=0)
    ap.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--num_retries", type=int, default=2)
    ap.add_argument(
        "--prompt_style",
        default="csv",
        choices=["csv", "tablegpt"],
        help="How to format the visible table in the prompt.",
    )
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
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def canonical_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [canonical_jsonable(v) for v in value]
    return canonical_value(value)


def canonical_payload_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
                return json.dumps(canonical_jsonable(parsed), ensure_ascii=False, sort_keys=True)
            except Exception:
                pass
    return json.dumps(canonical_jsonable(value), ensure_ascii=False, sort_keys=True)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return json.loads(text)
            except Exception:
                return text
    return value


def exact_match(predicted: Any, actual: Any) -> float:
    return 1.0 if canonical_payload_text(predicted) == canonical_payload_text(actual) else 0.0


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def extract_payload_values(payload: Any) -> List[str]:
    parsed = parse_jsonish(payload)
    values: List[str] = []
    if isinstance(parsed, dict):
        for value in parsed.values():
            values.append(json.dumps(canonical_jsonable(value), ensure_ascii=False, sort_keys=True))
            if isinstance(value, list):
                values.extend(str(canonical_value(item)) for item in value)
            else:
                values.append(str(canonical_value(value)))
    elif isinstance(parsed, list):
        for value in parsed:
            values.append(str(canonical_value(value)))
    else:
        values.append(str(canonical_value(parsed)))
    out: List[str] = []
    seen = set()
    for value in values:
        norm = normalize_text(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(value)
    return out


def value_match(predicted: Any, actual_values: Sequence[Any]) -> float:
    norm_pred = normalize_text(predicted)
    if not norm_pred:
        return 0.0
    for value in actual_values:
        if norm_pred == normalize_text(value):
            return 1.0
    return 0.0


def fetch_full_table(conn: sqlite3.Connection, table_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    cur = conn.execute(f'SELECT rowid AS "__rowid__", * FROM "{table_name}"')
    cols = [desc[0] for desc in cur.description]
    rows = []
    for raw in cur.fetchall():
        rows.append({cols[idx]: canonical_value(raw[idx]) for idx in range(len(cols))})
    return cols, rows


def split_candidate_columns(text: str, schema_cols: Sequence[str]) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if raw in schema_cols:
        return [raw]
    pieces = [part.strip() for part in raw.split(" | ") if part.strip()]
    return [piece for piece in pieces if piece in schema_cols]


def choose_hidden_columns(
    *,
    annotated_row: Dict[str, str],
    schema_cols: Sequence[str],
    source_column: str,
    extra_hidden_columns: Sequence[str],
) -> List[str]:
    hidden: List[str] = []
    seen = set()
    for field_name in ("ground_truth_column", "final_column"):
        for col in split_candidate_columns(annotated_row.get(field_name, ""), schema_cols):
            if col == source_column:
                continue
            if col not in seen:
                seen.add(col)
                hidden.append(col)
    for col in extra_hidden_columns:
        if col in schema_cols and col != source_column and col not in seen:
            seen.add(col)
            hidden.append(col)
    return hidden


def drop_columns(rows: Sequence[Dict[str, Any]], hidden_columns: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    hidden = set(hidden_columns)
    if not rows:
        return [], []
    visible_cols = [col for col in rows[0].keys() if col not in hidden]
    visible_rows: List[Dict[str, Any]] = []
    for row in rows:
        visible_rows.append({col: row.get(col) for col in visible_cols})
    return visible_cols, visible_rows


def table_to_csv_text(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    if not columns:
        return ""
    parts = [",".join(csv_quote(col) for col in columns)]
    for row in rows:
        parts.append(",".join(csv_quote(row.get(col, "")) for col in columns))
    return "\n".join(parts)


def table_to_pandas_like_text(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    if not columns:
        return ""
    rendered_rows: List[List[str]] = []
    for row in rows:
        rendered_rows.append(["" if row.get(col) is None else str(row.get(col)) for col in columns])
    widths: List[int] = []
    for idx, col in enumerate(columns):
        max_cell = max([len(col)] + [len(r[idx]) for r in rendered_rows]) if rendered_rows else len(col)
        widths.append(max_cell)
    lines = []
    lines.append(" ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns)))
    for row in rendered_rows:
        lines.append(" ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def csv_quote(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in [",", "\"", "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def build_prompt(*, question: str, table_csv_text: str) -> str:
    return "\n".join(
        [
            "You are answering a table question using only the visible CSV table below.",
            'The column "__rowid__" uniquely identifies each visible row.',
            "Use only the visible table content. Do not use outside knowledge.",
            "Return ONLY valid JSON in this exact shape:",
            '{"predictions":[{"table_row_id": 1, "answer": ...}]}',
            "Return one prediction for each row that satisfies the question.",
            "Use the integer __rowid__ values from the table as table_row_id.",
            "If no rows satisfy the question, return {\"predictions\":[]}.",
            "The answer may be a string, boolean, number, list, or object.",
            "Do not add explanations.",
            "",
            f"Question: {question}",
            "",
            "Visible table (CSV):",
            table_csv_text,
        ]
    ).strip()


def build_tablegpt_prompt(*, question: str, table_text: str) -> str:
    return "\n".join(
        [
            "Given access to a pandas-like dataframe, answer the user's question using only the visible table below.",
            'The column "__rowid__" uniquely identifies each visible row.',
            "Do not use outside knowledge.",
            "Return ONLY valid JSON in this exact shape:",
            '{"predictions":[{"table_row_id": 1, "answer": ...}]}',
            "Return one prediction for each row that satisfies the question.",
            "Use the integer __rowid__ values from the table as table_row_id.",
            "If no rows satisfy the question, return {\"predictions\":[]}.",
            "The answer may be a string, boolean, number, list, or object.",
            "Do not add explanations.",
            "",
            "df.to_string(index=False) as follows:",
            table_text,
            "",
            f"Question: {question}",
        ]
    ).strip()


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
                    "naive_llm call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    str(exc).splitlines()[0] if str(exc) else "",
                )
                raise
            backoff = min(30.0, (2 ** (attempt - 1)))
            logger.warning(
                "Retrying naive_llm call attempt=%d/%d after %.1fs (err=%s)",
                attempt,
                attempts,
                backoff,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            time.sleep(backoff)


def parse_predictions(raw_text: str) -> List[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    data: Any
    try:
        data = json.loads(text)
    except Exception:
        start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start < 0 or end < start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return []

    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            data = data["predictions"]
        else:
            data = [data]
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in data:
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
        out.append(
            {
                "table_row_id": row_id_int,
                "answer": canonical_jsonable(answer),
            }
        )
    return out


def visible_gt_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(canonical_value(row.get("NCT", "") or "")),
        str(canonical_value(row.get("PubMed ID", "") or "")),
        str(canonical_value(row.get("Trial name", "") or "")),
        str(canonical_value(row.get("source_value", "") or "")),
    )


def visible_table_key(row: Dict[str, Any], source_column: str) -> Tuple[str, str, str, str]:
    return (
        str(canonical_value(row.get("NCT", "") or "")),
        str(canonical_value(row.get("PubMed ID", "") or "")),
        str(canonical_value(row.get("Trial name", "") or "")),
        str(canonical_value(row.get(source_column, "") or "")),
    )


def assign_ground_truth_row_ids(
    *,
    full_table_rows: Sequence[Dict[str, Any]],
    source_column: str,
    ground_truth_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    queues: DefaultDict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for row in full_table_rows:
        key = visible_table_key(row, source_column)
        row_id = row.get("__rowid__")
        if row_id is None:
            continue
        try:
            queues[key].append(int(row_id))
        except Exception:
            continue

    assigned: List[Dict[str, Any]] = []
    for gt_row in ground_truth_rows:
        key = visible_gt_key(gt_row)
        row_id: Optional[int] = None
        if queues[key]:
            row_id = queues[key].pop(0)
        item = dict(gt_row)
        item["table_row_id"] = row_id
        assigned.append(item)
    return assigned


def load_sample_questions(
    *,
    manifest_csv: Path,
    annotated_csv: Path,
    limit: int,
    csv_row_number: int,
) -> List[Dict[str, Any]]:
    manifest_rows = read_csv_rows(manifest_csv)
    annotated_rows = read_csv_rows(annotated_csv)
    out: List[Dict[str, Any]] = []
    for manifest_row in manifest_rows:
        if (manifest_row.get("status") or "").strip() != "ok":
            continue
        row_number = int(manifest_row["csv_row_number"])
        if csv_row_number and row_number != csv_row_number:
            continue
        annotated_row = annotated_rows[row_number - 2]
        question_dir = Path(manifest_row["question_dir"])
        metadata = read_json(question_dir / "metadata.json")
        gt_rows = read_csv_rows(question_dir / "ground_truth_table.csv")
        out.append(
            {
                "item_id": manifest_row["item_id"],
                "csv_row_number": row_number,
                "question_dir": str(question_dir),
                "question": metadata["question"],
                "column_used": metadata["column_used"],
                "expected_keys": list(metadata["expected_keys"]),
                "ground_truth_rows": gt_rows,
                "annotated_row": annotated_row,
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    annotated_csv = Path(args.annotated_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip() or "question_table_copy_llm_naive_100")
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"qtable_llm_{run_dir.name}")

    logger.info("Manifest CSV: %s", manifest_csv)
    logger.info("Annotated CSV: %s", annotated_csv)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Model: %s", args.model_name)

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

    for item in sample_questions:
        question = item["question"]
        source_column = item["column_used"]
        if source_column not in table_cols:
            run_counter["missing_source_column"] += 1
            continue

        hidden_columns = choose_hidden_columns(
            annotated_row=item["annotated_row"],
            schema_cols=table_cols,
            source_column=source_column,
            extra_hidden_columns=args.answer_leak_columns,
        )
        visible_cols, visible_rows = drop_columns(full_table_rows, hidden_columns)
        if args.prompt_style == "tablegpt":
            table_text = table_to_pandas_like_text(visible_cols, visible_rows)
            prompt = build_tablegpt_prompt(question=question, table_text=table_text)
        else:
            table_text = table_to_csv_text(visible_cols, visible_rows)
            prompt = build_prompt(question=question, table_csv_text=table_text)

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
        (question_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        requests_rows.append(
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "hidden_columns_json": json.dumps(hidden_columns, ensure_ascii=False),
                "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                "prompt_char_count": len(prompt),
                "prompt_style": args.prompt_style,
                "table_row_count": len(visible_rows),
                "table_column_count": len(visible_cols),
                "prompt": prompt,
            }
        )

        raw_text = ""
        parsed_predictions: List[Dict[str, Any]] = []
        error = ""
        model_meta: Dict[str, Any] = {}
        try:
            resp_obj = run_one_call_with_retries(
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
            )
            raw_text = parse_chat_completion_text(resp_obj)
            parsed_predictions = parse_predictions(raw_text)
            model_meta = completion_meta(resp_obj)
        except Exception as exc:
            error = str(exc)

        responses_rows.append(
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "llm_raw_output": raw_text,
                "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                "error": error,
                "model_meta_json": json.dumps(model_meta, ensure_ascii=False),
            }
        )

        pred_map: Dict[int, Any] = {}
        for pred in parsed_predictions:
            row_id = pred["table_row_id"]
            if row_id not in pred_map:
                pred_map[row_id] = pred["answer"]

        question_row_start = len(row_rows)
        gt_row_ids = [row.get("table_row_id") for row in assigned_gt_rows if row.get("table_row_id") is not None]
        gt_row_id_set = {int(row_id) for row_id in gt_row_ids}
        pred_row_id_set = set(pred_map.keys())
        matched_row_ids = gt_row_id_set & pred_row_id_set

        matched_exact = 0
        row_level_count = 0
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
            row_rows.append(
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
            row_rows.append(
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

        question_rows.append(
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                "hidden_columns_json": json.dumps(hidden_columns, ensure_ascii=False),
                "table_row_count": len(visible_rows),
                "table_column_count": len(visible_cols),
                "ground_truth_row_count": len(item["ground_truth_rows"]),
                "assigned_ground_truth_row_count": row_level_count,
                "predicted_row_count": pred_count,
                "row_selection_recall": row_recall,
                "row_selection_precision": row_precision,
                "row_selection_f1": row_f1,
                "exact_match_rate": exact_match_rate,
                "all_rows_exact_match": all_rows_exact_match,
                "llm_raw_output": raw_text,
                "parsed_predictions_json": json.dumps(parsed_predictions, ensure_ascii=False),
                "error": error,
                "model_meta_json": json.dumps(model_meta, ensure_ascii=False),
                "question_dir": str(question_dir),
                "table_copy_csv": str(table_copy_path),
                "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
                "ground_truth_with_rowids_csv": str(gt_with_rowids_csv),
                "prompt_txt": str(question_dir / "prompt.txt"),
            }
        )

        write_csv(question_dir / "llm_predictions.csv", row_rows[question_row_start:])
        write_json(
            question_dir / "metadata.json",
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "expected_keys": item["expected_keys"],
                "hidden_columns": hidden_columns,
                "table_copy_csv": str(table_copy_path),
                "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
                "ground_truth_with_rowids_csv": str(gt_with_rowids_csv),
                "prompt_txt": str(question_dir / "prompt.txt"),
                "llm_raw_output": raw_text,
                "parsed_predictions": parsed_predictions,
                "error": error,
                "model_meta": model_meta,
            },
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
            "avg_row_selection_recall": avg_numeric(question_rows, "row_selection_recall"),
            "avg_row_selection_precision": avg_numeric(question_rows, "row_selection_precision"),
            "avg_row_selection_f1": avg_numeric(question_rows, "row_selection_f1"),
            "avg_exact_match_rate": avg_numeric(question_rows, "exact_match_rate"),
            "avg_all_rows_exact_match": avg_numeric(question_rows, "all_rows_exact_match"),
        }
    ]
    baseline_summary_csv = run_dir / "baseline_summary.csv"
    write_csv(baseline_summary_csv, summary_rows)

    summary = {
        "manifest_csv": str(manifest_csv),
        "annotated_csv": str(annotated_csv),
        "db_path": str(db_path),
        "model_name": args.model_name,
        "sample_question_count": len(sample_questions),
        "question_result_count": len(question_rows),
        "row_result_count": len(row_rows),
        "skips": dict(run_counter),
        "outputs": {
            "sample_manifest_csv": str(run_dir / "sample_manifest.csv"),
            "question_results_csv": str(question_results_csv),
            "row_level_predictions_csv": str(row_level_csv),
            "requests_csv": str(requests_csv),
            "responses_csv": str(responses_csv),
            "baseline_summary_csv": str(baseline_summary_csv),
        },
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote sample manifest: %s", run_dir / "sample_manifest.csv")
    logger.info("Wrote question results: %s", question_results_csv)
    logger.info("Wrote row-level predictions: %s", row_level_csv)
    logger.info("Wrote summary: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
