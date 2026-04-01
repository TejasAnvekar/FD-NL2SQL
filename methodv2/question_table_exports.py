#!/usr/bin/env python3
"""Helpers for exporting per-question comparison tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence


def _slugify(text: str, max_chars: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "question"
    return slug[:max_chars].rstrip("_") or "question"


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _collapse_actual_values(values: Sequence[Any]) -> str:
    rendered = [_stringify_cell(value) for value in values if value is not None]
    if not rendered:
        return ""
    return " | ".join(rendered)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify_cell(row.get(key)) for key in fieldnames})


def export_question_tables(
    *,
    questions_root_dir: Path,
    item_id: str,
    question: str,
    hidden_column: str,
    group_name: str,
    prompt_mode: str,
    gt_sql: str,
    pred_sql: str,
    gt_visible_cols: Sequence[str],
    gt_visible_rows: Sequence[Dict[str, Any]],
    pred_visible_cols: Sequence[str],
    pred_visible_rows: Sequence[Dict[str, Any]],
    pred_by_row_index: Dict[int, Any],
    actual_mapping: Dict[str, Dict[str, Any]],
    row_key_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, str]:
    question_dir = questions_root_dir / f"{group_name}__{item_id}__{_slugify(question)}"
    question_dir.mkdir(parents=True, exist_ok=True)

    final_fieldnames = ["row_index"] + list(pred_visible_cols) + [hidden_column]
    if len(final_fieldnames) == 2:
        final_fieldnames = ["row_index", hidden_column]
    final_rows: List[Dict[str, Any]] = []
    for row_index, row_obj in enumerate(pred_visible_rows, start=1):
        row_out: Dict[str, Any] = {"row_index": row_index}
        row_out.update(row_obj)
        row_out[hidden_column] = pred_by_row_index.get(row_index)
        final_rows.append(row_out)
    _write_csv(question_dir / "final_table.csv", final_fieldnames, final_rows)

    gt_fieldnames = ["row_index"] + list(gt_visible_cols) + [hidden_column]
    if len(gt_fieldnames) == 2:
        gt_fieldnames = ["row_index", hidden_column]
    gt_rows_out: List[Dict[str, Any]] = []
    for row_index, row_obj in enumerate(gt_visible_rows, start=1):
        actual_entry = actual_mapping.get(row_key_fn(row_obj), {"actual_values": []})
        row_out = {"row_index": row_index}
        row_out.update(row_obj)
        row_out[hidden_column] = _collapse_actual_values(actual_entry.get("actual_values") or [])
        gt_rows_out.append(row_out)
    _write_csv(question_dir / "ground_truth_table.csv", gt_fieldnames, gt_rows_out)

    metadata = {
        "item_id": item_id,
        "question": question,
        "hidden_column": hidden_column,
        "group_name": group_name,
        "prompt_source_column_mode": prompt_mode,
        "gt_sql": gt_sql,
        "pred_sql": pred_sql,
        "final_table_csv": str(question_dir / "final_table.csv"),
        "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
    }
    with (question_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {
        "question_dir": str(question_dir),
        "final_table_csv": str(question_dir / "final_table.csv"),
        "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv"),
        "metadata_json": str(question_dir / "metadata.json"),
    }
