#!/usr/bin/env python3
"""Cosine-retrieval baselines over exported table-derivation ground truths.

This runner operates on question-level exports from
`data/table_question_ground_truths_full`.

For each sampled question:
1. Read its `ground_truth_table.csv` and `metadata.json`.
2. Build visible evidence rows from `NCT`, `PubMed ID`, `Trial name`, and
   `source_value`.
3. Predict the derived answer via leave-one-out nearest-neighbor cosine
   retrieval inside the same expected-key family.

Views:
- column: source_value only
- row: visible row only
- tuple: question + visible row
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_embedding_cosine_baselines import (  # noqa: E402
    DEFAULT_BERT_MODEL,
    DEFAULT_BGE_MODEL,
    DEFAULT_SBERT_MODEL,
    LocalEmbedder,
    avg_numeric,
    build_column_text,
    build_row_text,
    build_tuple_text,
    cosine_top1,
    write_csv,
)
from run_hidden_column_sql_eval import make_run_dir  # noqa: E402
from utils import setup_logger, write_json  # noqa: E402

DEFAULT_MANIFEST = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv"
DEFAULT_RUN_ROOT = "/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run cosine baselines over exported table-derivation question ground truths. "
            "A 100-question sample is taken from the manifest, and leave-one-out nearest-neighbor "
            "retrieval predicts the derived answer."
        )
    )
    ap.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    ap.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--run_name", default="question_ground_truth_cosine_100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--baseline_models",
        nargs="+",
        default=["bert", "sbert", "bge"],
        choices=["bert", "sbert", "bge"],
    )
    ap.add_argument("--bert_model_path", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--sbert_model_path", default=DEFAULT_SBERT_MODEL)
    ap.add_argument("--bge_model_path", default=DEFAULT_BGE_MODEL)
    return ap.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [canonical_jsonable(item) for item in value]
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


def canonical_payload_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
                return json.dumps(canonical_jsonable(parsed), ensure_ascii=False, sort_keys=True)
            except Exception:
                pass
        return json.dumps(canonical_jsonable(text), ensure_ascii=False, sort_keys=True)
    return json.dumps(canonical_jsonable(value), ensure_ascii=False, sort_keys=True)


def exact_match(predicted_payload: str, actual_payload: str) -> float:
    return 1.0 if canonical_payload_text(predicted_payload) == canonical_payload_text(actual_payload) else 0.0


def visible_row_from_gt_row(row: Dict[str, str]) -> Dict[str, Any]:
    visible_cols = ["NCT", "PubMed ID", "Trial name", "source_value"]
    return {col: row.get(col, "") for col in visible_cols if col in row}


def key_signature(expected_keys: Sequence[str]) -> str:
    return "|".join(sorted(str(key) for key in expected_keys))


def model_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for name in args.baseline_models:
        if name == "bert":
            specs.append(("bert", args.bert_model_path, ""))
        elif name == "sbert":
            specs.append(("sbert", args.sbert_model_path, ""))
        elif name == "bge":
            specs.append(("bge", args.bge_model_path, "Represent this sentence for retrieval: "))
    return specs


def load_sample_questions(manifest_csv: Path, limit: int) -> List[Dict[str, Any]]:
    manifest_rows = read_csv_rows(manifest_csv)
    questions: List[Dict[str, Any]] = []
    for manifest_row in manifest_rows:
        if (manifest_row.get("status") or "").strip() != "ok":
            continue
        question_dir = Path(manifest_row["question_dir"])
        metadata = read_json(question_dir / "metadata.json")
        gt_rows = read_csv_rows(question_dir / "ground_truth_table.csv")
        questions.append(
            {
                "item_id": manifest_row["item_id"],
                "csv_row_number": int(manifest_row["csv_row_number"]),
                "question_dir": str(question_dir),
                "question": metadata["question"],
                "column_used": metadata["column_used"],
                "expected_keys": list(metadata["expected_keys"]),
                "key_signature": key_signature(metadata["expected_keys"]),
                "ground_truth_rows": gt_rows,
            }
        )
        if limit and len(questions) >= limit:
            break
    return questions


def build_dataset(sample_questions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for question in sample_questions:
        for row in question["ground_truth_rows"]:
            row_index = int(row.get("row_index") or len(dataset) + 1)
            visible_row = visible_row_from_gt_row(row)
            dataset.append(
                {
                    "dataset_id": f"{question['item_id']}::row_{row_index}",
                    "item_id": question["item_id"],
                    "csv_row_number": question["csv_row_number"],
                    "question": question["question"],
                    "column_used": question["column_used"],
                    "key_signature": question["key_signature"],
                    "expected_keys": list(question["expected_keys"]),
                    "row_index": row_index,
                    "visible_row": visible_row,
                    "source_value": row.get("source_value", ""),
                    "target_payload": row.get("derived_expected_llm_response", ""),
                    "question_dir": question["question_dir"],
                }
            )
    return dataset


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = make_run_dir(run_root, args.run_name.strip() or "question_ground_truth_cosine_100")
    logger = setup_logger(str(run_dir / "logs"), str(run_dir / "run_meta.json"), logger_name=f"qgt_cosine_{run_dir.name}")

    logger.info("Manifest CSV: %s", manifest_csv)
    logger.info("Run dir: %s", run_dir)

    sample_questions = load_sample_questions(manifest_csv, args.limit)
    dataset = build_dataset(sample_questions)
    key_counts = Counter(question["key_signature"] for question in sample_questions)

    sample_manifest_rows = [
        {
            "item_id": question["item_id"],
            "csv_row_number": question["csv_row_number"],
            "question": question["question"],
            "column_used": question["column_used"],
            "expected_keys_json": json.dumps(question["expected_keys"], ensure_ascii=False),
            "key_signature": question["key_signature"],
            "question_dir": question["question_dir"],
            "ground_truth_row_count": len(question["ground_truth_rows"]),
        }
        for question in sample_questions
    ]
    write_csv(run_dir / "sample_manifest.csv", sample_manifest_rows)

    meta = {
        "manifest_csv": str(manifest_csv),
        "sample_question_count": len(sample_questions),
        "sample_row_count": len(dataset),
        "key_signature_counts": dict(sorted(key_counts.items())),
        "baseline_models": list(args.baseline_models),
        "device": args.device,
    }
    write_json(run_dir / "run_meta.json", meta)

    model_specs = model_specs_from_args(args)
    embedders: Dict[str, LocalEmbedder] = {}
    for name, model_path, instruction_prefix in model_specs:
        logger.info("Loading embedder %s from %s", name, model_path)
        embedders[name] = LocalEmbedder(model_path, device=args.device, instruction_prefix=instruction_prefix)

    grouped_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in dataset:
        grouped_dataset[item["key_signature"]].append(item)

    row_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []

    for key_sig, group_items in sorted(grouped_dataset.items()):
        logger.info("Scoring key family %s (%d rows)", key_sig, len(group_items))
        if len(group_items) < 2:
            logger.info("Skipping key family %s because it has fewer than 2 rows", key_sig)
            continue

        column_texts = [build_column_text(item["source_value"]) for item in group_items]
        row_texts = [build_row_text(item["visible_row"]) for item in group_items]
        tuple_texts = [build_tuple_text(item["question"], item["visible_row"]) for item in group_items]

        for model_name, embedder in embedders.items():
            column_matrix = embedder.encode(column_texts, batch_size=args.batch_size)
            row_matrix = embedder.encode(row_texts, batch_size=args.batch_size)
            tuple_matrix = embedder.encode(tuple_texts, batch_size=args.batch_size)

            for idx, item in enumerate(group_items):
                per_item_metrics: Dict[str, Dict[str, Any]] = {}
                for view_name, matrix in (
                    ("column", column_matrix),
                    ("row", row_matrix),
                    ("tuple", tuple_matrix),
                ):
                    query_vec = matrix[idx]
                    scores = np.matmul(matrix, query_vec)
                    scores[idx] = -np.inf  # leave-one-out
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    predicted_payload = group_items[best_idx]["target_payload"]
                    actual_payload = item["target_payload"]
                    exact = exact_match(predicted_payload, actual_payload)

                    row_record = {
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": item["question"],
                        "column_used": item["column_used"],
                        "key_signature": item["key_signature"],
                        "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                        "row_index": item["row_index"],
                        "model_name": model_name,
                        "view_name": view_name,
                        "visible_row_json": json.dumps(item["visible_row"], ensure_ascii=False, sort_keys=True),
                        "source_value": item["source_value"],
                        "actual_payload_json": actual_payload,
                        "predicted_payload_json": predicted_payload,
                        "neighbor_item_id": group_items[best_idx]["item_id"],
                        "neighbor_csv_row_number": group_items[best_idx]["csv_row_number"],
                        "neighbor_row_index": group_items[best_idx]["row_index"],
                        "neighbor_visible_row_json": json.dumps(group_items[best_idx]["visible_row"], ensure_ascii=False, sort_keys=True),
                        "cosine_score": best_score,
                        "exact_match": exact,
                    }
                    row_rows.append(row_record)
                    per_item_metrics[view_name] = {
                        "cosine_score": best_score,
                        "exact_match": exact,
                        "predicted_payload_json": predicted_payload,
                        "neighbor_item_id": group_items[best_idx]["item_id"],
                    }

                question_rows.append(
                    {
                        "item_id": item["item_id"],
                        "csv_row_number": item["csv_row_number"],
                        "question": item["question"],
                        "column_used": item["column_used"],
                        "key_signature": item["key_signature"],
                        "expected_keys_json": json.dumps(item["expected_keys"], ensure_ascii=False),
                        "row_index": item["row_index"],
                        "model_name": model_name,
                        "question_dir": item["question_dir"],
                        "visible_row_json": json.dumps(item["visible_row"], ensure_ascii=False, sort_keys=True),
                        "actual_payload_json": item["target_payload"],
                        "metrics_json": json.dumps(per_item_metrics, ensure_ascii=False),
                    }
                )

    write_csv(run_dir / "row_level_predictions.csv", row_rows)
    write_csv(run_dir / "all_question_results.csv", question_rows)

    summary_rows: List[Dict[str, Any]] = []
    by_model_view: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in row_rows:
        by_model_view[(str(row["model_name"]), str(row["view_name"]))].append(row)
    for (model_name, view_name), rows_out in sorted(by_model_view.items()):
        summary_rows.append(
            {
                "model_name": model_name,
                "view_name": view_name,
                "row_count": len(rows_out),
                "avg_exact_match": avg_numeric(rows_out, "exact_match"),
                "avg_cosine_score": avg_numeric(rows_out, "cosine_score"),
            }
        )
    write_csv(run_dir / "baseline_summary.csv", summary_rows)

    summary = {
        "manifest_csv": str(manifest_csv),
        "sample_question_count": len(sample_questions),
        "sample_row_count": len(dataset),
        "scored_row_count": len(row_rows),
        "outputs": {
            "sample_manifest_csv": str(run_dir / "sample_manifest.csv"),
            "question_results_csv": str(run_dir / "all_question_results.csv"),
            "row_predictions_csv": str(run_dir / "row_level_predictions.csv"),
            "baseline_summary_csv": str(run_dir / "baseline_summary.csv"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Wrote sample manifest: %s", run_dir / "sample_manifest.csv")
    logger.info("Wrote question results: %s", run_dir / "all_question_results.csv")
    logger.info("Wrote row predictions: %s", run_dir / "row_level_predictions.csv")
    logger.info("Wrote summary: %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
