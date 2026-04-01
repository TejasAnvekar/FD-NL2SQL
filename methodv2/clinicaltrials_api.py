#!/usr/bin/env python3
"""ClinicalTrials.gov API helpers for study-level metadata retrieval."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import parse as urllib_parse
from urllib import request as urllib_request


API_BASE = "https://clinicaltrials.gov/api/v2"


def fetch_study_by_nct(nct_id: str, timeout: float = 60.0) -> Dict[str, Any]:
    nct = (nct_id or "").strip()
    if not nct:
        raise ValueError("NCT id is required")
    url = f"{API_BASE}/studies/{urllib_parse.quote(nct)}"
    req = urllib_request.Request(url, headers={"Accept": "application/json"})
    with urllib_request.urlopen(req, timeout=float(timeout)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_get(obj: Dict[str, Any], *path: str) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def summarize_study(study_obj: Dict[str, Any]) -> Dict[str, Any]:
    protocol = study_obj.get("protocolSection") if isinstance(study_obj, dict) else {}
    identification = _safe_get(protocol or {}, "identificationModule") or {}
    status = _safe_get(protocol or {}, "statusModule") or {}
    design = _safe_get(protocol or {}, "designModule") or {}
    conditions_module = _safe_get(protocol or {}, "conditionsModule") or {}
    arms_module = _safe_get(protocol or {}, "armsInterventionsModule") or {}
    references_module = _safe_get(protocol or {}, "referencesModule") or {}

    arm_groups = []
    for arm in _as_list(arms_module.get("armGroups")):
        if not isinstance(arm, dict):
            continue
        arm_groups.append(
            {
                "label": arm.get("label"),
                "type": arm.get("type"),
                "description": arm.get("description"),
                "intervention_names": _as_list(arm.get("interventionNames")),
            }
        )

    interventions = []
    for intervention in _as_list(arms_module.get("interventions")):
        if not isinstance(intervention, dict):
            continue
        interventions.append(
            {
                "type": intervention.get("type"),
                "name": intervention.get("name"),
                "description": intervention.get("description"),
                "other_names": _as_list(intervention.get("otherNames")),
                "arm_group_labels": _as_list(intervention.get("armGroupLabels")),
            }
        )

    pmids = []
    for ref in _as_list(references_module.get("references")):
        if not isinstance(ref, dict):
            continue
        pmid = ref.get("pmid")
        if pmid:
            pmids.append(str(pmid))

    candidate_control_arms = []
    for arm in arm_groups:
        label_text = " ".join(str(arm.get(key) or "") for key in ("label", "type", "description")).lower()
        if any(marker in label_text for marker in ("control", "comparator", "placebo", "standard of care", "active comparator")):
            candidate_control_arms.append(arm)
        elif str(arm.get("type") or "").lower() in {"active_comparator", "placebo_comparator", "sham_comparator", "no_intervention"}:
            candidate_control_arms.append(arm)

    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "acronym": identification.get("acronym"),
        "organization": _safe_get(identification, "organization", "fullName"),
        "overall_status": status.get("overallStatus"),
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "study_first_post_date": status.get("studyFirstPostDateStruct", "date"),
        "last_update_posted": status.get("lastUpdatePostDateStruct", "date"),
        "phases": _as_list(design.get("phases")),
        "study_type": design.get("studyType"),
        "conditions": _as_list(conditions_module.get("conditions")),
        "keywords": _as_list(conditions_module.get("keywords")),
        "arm_groups": arm_groups,
        "interventions": interventions,
        "candidate_control_arms": candidate_control_arms,
        "reference_pmids": pmids,
    }
