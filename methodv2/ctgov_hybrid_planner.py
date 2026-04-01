#!/usr/bin/env python3
"""Hybrid CTGov planning helpers for question-specific study evidence selection."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence


AVAILABLE_FIELDS = [
    "brief_title",
    "official_title",
    "acronym",
    "organization",
    "overall_status",
    "phases",
    "study_type",
    "conditions",
    "keywords",
    "arm_groups",
    "interventions",
    "candidate_control_arms",
    "reference_pmids",
]

NESTED_LIST_FIELDS = {"arm_groups", "interventions", "candidate_control_arms"}

FIELD_DESCRIPTIONS = {
    "brief_title": "Short study title from ClinicalTrials.gov.",
    "official_title": "Long-form official study title.",
    "acronym": "Study acronym or short identifier.",
    "organization": "Lead sponsor or responsible organization.",
    "overall_status": "Recruitment or completion status.",
    "phases": "Study phase information.",
    "study_type": "Interventional or observational study type.",
    "conditions": "Study conditions or cancer types.",
    "keywords": "Supplementary study keywords.",
    "arm_groups": "Arm labels, arm types, and arm descriptions.",
    "interventions": "Named interventions with types and descriptions.",
    "candidate_control_arms": "Likely comparator/control arms derived from the CTGov record.",
    "reference_pmids": "PubMed IDs referenced by the study record.",
}

DEFAULT_FALLBACK_FIELDS = [
    "brief_title",
    "acronym",
    "conditions",
    "arm_groups",
    "interventions",
    "candidate_control_arms",
]


def default_ctgov_plan(notes: str = "") -> Dict[str, Any]:
    return {
        "need_ctgov_metadata": True,
        "relevant_fields": list(DEFAULT_FALLBACK_FIELDS),
        "focus": "",
        "notes": notes,
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _row_tokens(row_obj: Dict[str, Any]) -> List[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "trial",
        "trials",
        "study",
        "arm",
        "plus",
        "drug",
        "treatment",
        "regimen",
    }
    tokens = set()
    for value in row_obj.values():
        text = _stringify(value).lower()
        for token in re.findall(r"[a-z0-9][a-z0-9\-]+", text):
            if len(token) < 4 or token in stop:
                continue
            tokens.add(token)
    return sorted(tokens)


def build_ctgov_planning_prompt(
    *,
    question: str,
    target_column: str,
    result_rows: Sequence[Dict[str, Any]],
    prompt_source_column_mode: str,
    source_column: str,
    ctgov_bundle: Dict[str, Any],
) -> str:
    preview_rows: List[str] = []
    rows_with_metadata = list(ctgov_bundle.get("rows_with_metadata") or [])
    for row_obj, meta in zip(result_rows, rows_with_metadata):
        preview: Dict[str, Any] = {
            "row": row_obj,
            "nct": meta.get("nct"),
            "trial_name": meta.get("trial_name"),
        }
        summary = meta.get("ctgov_summary")
        if isinstance(summary, dict):
            preview["ctgov_preview"] = {
                "brief_title": summary.get("brief_title"),
                "acronym": summary.get("acronym"),
                "conditions": summary.get("conditions"),
                "candidate_control_arm_labels": [x.get("label") for x in (summary.get("candidate_control_arms") or []) if isinstance(x, dict)],
                "arm_group_labels": [x.get("label") for x in (summary.get("arm_groups") or [])[:6] if isinstance(x, dict)],
                "intervention_names": [x.get("name") for x in (summary.get("interventions") or [])[:8] if isinstance(x, dict)],
            }
        preview_rows.append(json.dumps(preview, ensure_ascii=False, sort_keys=True))

    lines = [
        "You are choosing which ClinicalTrials.gov metadata fields are most useful for answering a question.",
        "The final answer will be produced in a later step.",
        "Your job now is only to select the most relevant study metadata fields to extract for the target field that will be inferred later.",
        "Return ONLY valid JSON in this exact shape:",
        '{"need_ctgov_metadata": true, "relevant_fields": ["field_name"], "focus": "...", "notes": "..."}',
        f"Allowed field names: {AVAILABLE_FIELDS}",
        "Field descriptions:",
        *[f"- {field}: {FIELD_DESCRIPTIONS[field]}" for field in AVAILABLE_FIELDS],
        "",
        f"Question: {question}",
        f"Target field to infer later: {target_column}",
    ]
    if prompt_source_column_mode == "shown":
        lines.append(f'Source column shown to the model: "{source_column}"')
    lines.extend(
        [
            "",
            "Few-shot examples:",
            'Example 1',
            'Question: For Atezolizumab combination therapy trials, list additional agents in the treatment regimen beyond the ICI.',
            'Target field to infer later: Control regimen',
            'Output: {"need_ctgov_metadata": true, "relevant_fields": ["candidate_control_arms", "arm_groups", "interventions"], "focus": "identify the comparator or control arm regimen for each study and normalize it to the control regimen value", "notes": "The question retrieves the studies, but the target field is Control regimen, so comparator-arm metadata is needed."}',
            "",
            'Example 2',
            'Question: In NSCLC trials classify the ICI by target class from its generic name.',
            'Target field to infer later: Class of ICI',
            'Output: {"need_ctgov_metadata": false, "relevant_fields": [], "focus": "the row evidence already names the ICI and CTGov metadata is not needed", "notes": "Do not request CTGov metadata when the target can be inferred directly from the visible row."}',
        ]
    )
    lines.append("Rows and compact CTGov previews:")
    lines.extend(preview_rows or ['{"rows": []}'])
    return "\n".join(lines).strip()


def parse_ctgov_plan(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return default_ctgov_plan("empty planner output")
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return default_ctgov_plan("planner parse failed")
        try:
            data = json.loads(match.group(0))
        except Exception:
            return default_ctgov_plan("planner parse failed")

    if not isinstance(data, dict):
        data = {}
    relevant_fields = [field for field in data.get("relevant_fields") or [] if field in AVAILABLE_FIELDS]
    if not relevant_fields:
        relevant_fields = list(DEFAULT_FALLBACK_FIELDS)
    return {
        "need_ctgov_metadata": bool(data.get("need_ctgov_metadata", True)),
        "relevant_fields": relevant_fields,
        "focus": str(data.get("focus") or ""),
        "notes": str(data.get("notes") or ""),
    }


def _filter_nested_items(items: Sequence[Dict[str, Any]], row_obj: Dict[str, Any], max_items: int = 6) -> List[Dict[str, Any]]:
    row_tokens = _row_tokens(row_obj)
    if not items:
        return []
    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for idx, item in enumerate(items):
        text = _stringify(item).lower()
        score = sum(1 for token in row_tokens if token in text)
        if any(marker in text for marker in ("control", "comparator", "placebo", "standard of care", "active comparator")):
            score += 3
        scored.append((score, -idx, item))
    scored.sort(reverse=True)
    selected = [item for score, _, item in scored if score > 0][:max_items]
    if not selected:
        selected = [item for _, _, item in scored[:max_items]]
    return selected


def select_ctgov_evidence_for_prompt(
    *,
    result_rows: Sequence[Dict[str, Any]],
    ctgov_bundle: Dict[str, Any],
    plan: Dict[str, Any],
    max_items_per_field: int = 6,
) -> List[Dict[str, Any]]:
    rows_with_metadata = list(ctgov_bundle.get("rows_with_metadata") or [])
    relevant_fields = list(plan.get("relevant_fields") or [])
    out: List[Dict[str, Any]] = []
    for row_index, row_obj in enumerate(result_rows, start=1):
        meta = rows_with_metadata[row_index - 1] if row_index - 1 < len(rows_with_metadata) else {}
        row_payload: Dict[str, Any] = {
            "row_index": row_index,
            "nct": meta.get("nct"),
            "trial_name": meta.get("trial_name"),
        }
        summary = meta.get("ctgov_summary")
        if isinstance(summary, dict) and bool(plan.get("need_ctgov_metadata", True)):
            selected: Dict[str, Any] = {}
            for field in relevant_fields:
                value = summary.get(field)
                if field in NESTED_LIST_FIELDS and isinstance(value, list):
                    filtered_items = [x for x in value if isinstance(x, dict)]
                    selected[field] = _filter_nested_items(filtered_items, row_obj, max_items=max_items_per_field)
                elif isinstance(value, list):
                    selected[field] = value[:max_items_per_field]
                else:
                    selected[field] = value
            row_payload["selected_ctgov_evidence"] = selected
        if meta.get("ctgov_error"):
            row_payload["ctgov_error"] = meta.get("ctgov_error")
        out.append(row_payload)
    return out
