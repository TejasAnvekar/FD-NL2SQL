#!/usr/bin/env python3
"""Strict TAPAS baseline using only a copied table plus the question.

For each sampled question:
1. Save a per-question copy of the full clinical_trials table.
2. Build a smaller TAPAS view with NCT, PubMed ID, Trial name, and source_value.
3. Run TAPAS chunk-by-chunk over the table view with the question.
4. Compare the selected cell answer against the derived ground truths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import TapasForQuestionAnswering, TapasTokenizer  # noqa: E402

from run_embedding_cosine_baselines import avg_numeric, write_csv  # noqa: E402
from run_hidden_column_sql_eval import make_run_dir, sanitize_name  # noqa: E402
from utils import setup_logger, write_json  # noqa: E402

DEFAULT_MANIFEST = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv"
DEFAULT_DB_PATH = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"
DEFAULT_MODEL = "google/tapas-base-finetuned-wtq"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a strict TAPAS baseline on table-derivation questions."
    )
    ap.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="question_table_copy_tapas_100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--max_rows_per_chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def exact_match(predicted: Any, actual: Any) -> float:
    return 1.0 if canonical_payload_text(predicted) == canonical_payload_text(actual) else 0.0


def value_match(predicted_text: str, actual_values: Sequence[Any]) -> float:
    norm_pred = normalize_text(predicted_text)
    if not norm_pred:
        return 0.0
    for value in actual_values:
        if norm_pred == normalize_text(value):
            return 1.0
    return 0.0


def extract_payload_values(payload_text: str) -> List[str]:
    text = (payload_text or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return [text]
    if isinstance(payload, dict):
        values: List[str] = []
        for value in payload.values():
            values.append(json.dumps(canonical_jsonable(value), ensure_ascii=False, sort_keys=True))
            if isinstance(value, list):
                values.extend(str(canonical_value(item)) for item in value)
            else:
                values.append(str(canonical_value(value)))
        return list(dict.fromkeys(values))
    return [json.dumps(canonical_jsonable(payload), ensure_ascii=False, sort_keys=True)]


def fetch_full_table(conn: sqlite3.Connection, table_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    cur = conn.execute(f'SELECT rowid AS "__rowid__", * FROM "{table_name}"')
    cols = [desc[0] for desc in cur.description]
    rows = []
    for raw in cur.fetchall():
        rows.append({cols[idx]: canonical_value(raw[idx]) for idx in range(len(cols))})
    return cols, rows


def load_sample_questions(manifest_csv: Path, limit: int) -> List[Dict[str, Any]]:
    rows = read_csv_rows(manifest_csv)
    out: List[Dict[str, Any]] = []
    for manifest_row in rows:
        if (manifest_row.get("status") or "").strip() != "ok":
            continue
        question_dir = Path(manifest_row["question_dir"])
        metadata = read_json(question_dir / "metadata.json")
        gt_rows = read_csv_rows(question_dir / "ground_truth_table.csv")
        out.append(
            {
                "item_id": manifest_row["item_id"],
                "csv_row_number": int(manifest_row["csv_row_number"]),
                "question_dir": str(question_dir),
                "question": metadata["question"],
                "column_used": metadata["column_used"],
                "expected_keys": list(metadata["expected_keys"]),
                "ground_truth_rows": gt_rows,
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def chunk_rows(rows: Sequence[Dict[str, Any]], chunk_size: int) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
    for start in range(0, len(rows), chunk_size):
        yield start, list(rows[start : start + chunk_size])


def chunk_score(logits: torch.Tensor, coords: Sequence[Tuple[int, int]]) -> float:
    if not coords:
        return float(logits.max().item())
    return float(logits.max().item())


def run_tapas_on_question(
    *,
    question: str,
    table_rows: Sequence[Dict[str, Any]],
    tokenizer: TapasTokenizer,
    model: TapasForQuestionAnswering,
    device: str,
    max_rows_per_chunk: int,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {
        "predicted_answer": "",
        "selected_cells": [],
        "aggregation_prediction": None,
        "chunk_score": float("-inf"),
        "chunk_start": 0,
    }

    for start, chunk in chunk_rows(table_rows, max_rows_per_chunk):
        if not chunk:
            continue
        df = pd.DataFrame(chunk).fillna("").astype(str)
        inputs = tokenizer(
            table=df,
            queries=[question],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        cpu_inputs = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        predicted_coords, predicted_agg = tokenizer.convert_logits_to_predictions(
            cpu_inputs,
            outputs.logits.detach().cpu(),
            outputs.logits_aggregation.detach().cpu(),
        )
        coords = predicted_coords[0]
        agg = int(predicted_agg[0]) if len(predicted_agg) else 0

        selected_cells: List[str] = []
        for row_idx, col_idx in coords:
            if row_idx < len(df.index) and col_idx < len(df.columns):
                selected_cells.append(str(df.iat[row_idx, col_idx]))
        selected_cells = list(dict.fromkeys(selected_cells))
        predicted_answer = " | ".join(cell for cell in selected_cells if str(cell).strip())
        score = chunk_score(outputs.logits.detach().cpu(), coords)

        if score > best["chunk_score"]:
            best = {
                "predicted_answer": predicted_answer,
                "selected_cells": selected_cells,
                "aggregation_prediction": agg,
                "chunk_score": score,
                "chunk_start": start,
                "chunk_row_count": len(chunk),
                "table_chunk_json": json.dumps(chunk, ensure_ascii=False, sort_keys=True),
            }

    return best


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip() or "question_table_copy_tapas_100")
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"qtapas_{run_dir.name}")

    logger.info("Manifest CSV: %s", manifest_csv)
    logger.info("DB path: %s", db_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Model: %s", args.model_name_or_path)

    sample_questions = load_sample_questions(manifest_csv, args.limit)
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

    tokenizer = TapasTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    model = TapasForQuestionAnswering.from_pretrained(args.model_name_or_path, local_files_only=True)
    model.eval()
    model.to(args.device)

    question_rows: List[Dict[str, Any]] = []

    for item in sample_questions:
        source_column = item["column_used"]
        if source_column not in table_cols:
            continue

        question = item["question"]
        question_slug = sanitize_name(f"{item['item_id']}__{question}")[:180]
        question_dir = run_dir / "questions" / question_slug
        question_dir.mkdir(parents=True, exist_ok=True)

        table_copy_path = question_dir / "table_copy.csv"
        write_csv(table_copy_path, full_table_rows)

        tapas_view_rows = []
        for row in full_table_rows:
            value = row.get(source_column)
            if value is None or str(value).strip() == "":
                continue
            tapas_view_rows.append(
                {
                    "NCT": row.get("NCT", ""),
                    "PubMed ID": row.get("PubMed ID", ""),
                    "Trial name": row.get("Trial name", ""),
                    "source_value": value,
                }
            )
        write_csv(question_dir / "tapas_table_view.csv", tapas_view_rows)
        write_csv(question_dir / "ground_truth_table.csv", item["ground_truth_rows"])

        gt_payloads = [row.get("derived_expected_llm_response", "") for row in item["ground_truth_rows"]]
        gt_values: List[str] = []
        gt_source_values: List[str] = []
        for row in item["ground_truth_rows"]:
            gt_values.extend(extract_payload_values(row.get("derived_expected_llm_response", "")))
            gt_source_values.append(str(row.get("source_value", "") or ""))
        gt_values = list(dict.fromkeys(v for v in gt_values if str(v).strip()))
        gt_source_values = list(dict.fromkeys(v for v in gt_source_values if str(v).strip()))

        predicted = run_tapas_on_question(
            question=question,
            table_rows=tapas_view_rows,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            max_rows_per_chunk=args.max_rows_per_chunk,
        )

        question_row = {
            "item_id": item["item_id"],
            "csv_row_number": item["csv_row_number"],
            "question": question,
            "column_used": source_column,
            "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
            "ground_truth_row_count": len(item["ground_truth_rows"]),
            "predicted_answer": predicted["predicted_answer"],
            "selected_cells_json": json.dumps(predicted["selected_cells"], ensure_ascii=False),
            "aggregation_prediction": predicted["aggregation_prediction"],
            "chunk_score": predicted["chunk_score"],
            "chunk_start": predicted["chunk_start"],
            "chunk_row_count": predicted.get("chunk_row_count", 0),
            "match_any_payload": max(exact_match(predicted["predicted_answer"], payload) for payload in gt_payloads) if gt_payloads else 0.0,
            "match_any_value": value_match(predicted["predicted_answer"], gt_values),
            "match_any_source_value": value_match(predicted["predicted_answer"], gt_source_values),
            "actual_payloads_json": json.dumps(gt_payloads, ensure_ascii=False),
            "actual_values_json": json.dumps(gt_values, ensure_ascii=False),
            "actual_source_values_json": json.dumps(gt_source_values, ensure_ascii=False),
            "question_dir": str(question_dir),
            "table_copy_csv": str(table_copy_path),
            "tapas_table_view_csv": str(question_dir / "tapas_table_view.csv"),
            "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
        }
        question_rows.append(question_row)

        write_json(
            question_dir / "metadata.json",
            {
                "item_id": item["item_id"],
                "csv_row_number": item["csv_row_number"],
                "question": question,
                "column_used": source_column,
                "expected_keys": item["expected_keys"],
                "predicted_answer": predicted["predicted_answer"],
                "selected_cells": predicted["selected_cells"],
                "aggregation_prediction": predicted["aggregation_prediction"],
                "chunk_score": predicted["chunk_score"],
                "chunk_start": predicted["chunk_start"],
                "table_copy_csv": str(table_copy_path),
                "tapas_table_view_csv": str(question_dir / "tapas_table_view.csv"),
                "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
            },
        )

    question_results_csv = run_dir / "all_question_results.csv"
    write_csv(question_results_csv, question_rows)

    summary_rows = [
        {
            "question_count": len(question_rows),
            "avg_match_any_payload": avg_numeric(question_rows, "match_any_payload"),
            "avg_match_any_value": avg_numeric(question_rows, "match_any_value"),
            "avg_match_any_source_value": avg_numeric(question_rows, "match_any_source_value"),
            "avg_chunk_score": avg_numeric(question_rows, "chunk_score"),
        }
    ]
    baseline_summary_csv = run_dir / "baseline_summary.csv"
    write_csv(baseline_summary_csv, summary_rows)

    summary = {
        "manifest_csv": str(manifest_csv),
        "db_path": str(db_path),
        "model_name_or_path": args.model_name_or_path,
        "sample_question_count": len(sample_questions),
        "question_result_count": len(question_rows),
        "outputs": {
            "sample_manifest_csv": str(run_dir / "sample_manifest.csv"),
            "question_results_csv": str(question_results_csv),
            "baseline_summary_csv": str(baseline_summary_csv),
        },
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote sample manifest: %s", run_dir / "sample_manifest.csv")
    logger.info("Wrote question results: %s", question_results_csv)
    logger.info("Wrote summary: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
