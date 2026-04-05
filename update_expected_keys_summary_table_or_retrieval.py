#!/usr/bin/env python3
"""Label expected-key summaries as Table vs Retrieval.

This operates on the unique-key summary CSV and applies a judgment-based split:
- Table: the answer can be derived from an existing table column by parsing,
  normalization, bucketing, boolean conversion, string construction, or
  semantic mapping from the visible column.
- Retrieval: the answer usually requires external trial-specific knowledge or
  publication/registry evidence that is not reliably available in the table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_expected_keys_summary.csv"


TABLE_RULES: Dict[str, Tuple[str, str]] = {
    "added_agents": ("Table", "Parse extra non-ICI drugs directly from Treatment regimen text."),
    "anti_vegf_agents": ("Table", "Identify anti-VEGF drugs directly from Treatment regimen text."),
    "arm_category": ("Table", "Bucket Number of arms into coarse categories directly from the numeric field."),
    "braf_inhibitor": ("Table", "Extract BRAFi agent names directly from Treatment regimen text."),
    "brand_name": ("Table", "Map generic ICI names to brand names via semantic drug-name understanding."),
    "class": ("Table", "Map generic ICI names to PD-1/PD-L1/CTLA-4 via semantic drug-class understanding."),
    "combo_type_norm": ("Table", "Normalize Type of combination into standardized labels from the table text."),
    "components": ("Table", "Split composite/co-primary endpoint text into its component endpoints."),
    "control_backbone": ("Table", "Derive chemotherapy backbone directly from Control regimen wording."),
    "control_category": ("Table", "Classify the Control arm text as placebo/active/BSC from table wording."),
    "control_drugs": ("Table", "Extract cytotoxic drug names directly from Control regimen text."),
    "endpoint_role": ("Table", "Determine endpoint role from whether the endpoint comes from the primary or secondary endpoint field."),
    "era": ("Table", "Bucket the Year column directly into predefined era ranges."),
    "followup_bucket": ("Table", "Bucket follow-up months directly from the follow-up duration column."),
    "followup_type": ("Table", "Normalize the follow-up type directly from Type of follow-up given."),
    "irAE_assessed": ("Table", "Flag irAE assessment directly from Secondary endpoint text."),
    "is_combination": ("Table", "Convert Type of therapy into a boolean combination flag."),
    "is_follow_up": ("Table", "Convert Original publication or Follow-up into a boolean follow-up flag."),
    "maturity": ("Table", "Classify maturity directly from follow-up months with a numeric rule."),
    "mek_inhibitor": ("Table", "Extract MEK inhibitor names directly from Treatment regimen text."),
    "meki_agents": ("Table", "Extract MEKi agents directly from Treatment regimen text."),
    "pd_l1_required": ("Table", "Convert the PD-L1 requirement field into boolean form."),
    "phase_bucket": ("Table", "Bucket Trial phase directly into early/late groups."),
    "platinum_agent": ("Table", "Identify cisplatin vs carboplatin directly from Treatment regimen text."),
    "primary_endpoint_category": ("Table", "Normalize Primary endpoint text into canonical endpoint categories."),
    "pubmed_url": ("Table", "Construct a PubMed URL deterministically from PubMed ID."),
    "schedule": ("Table", "Extract dosing schedule directly from Treatment regimen text."),
    "setting_normalized": ("Table", "Normalize Clinical setting into canonical labels using table wording."),
    "size_bucket": ("Table", "Bucket Total sample size directly from the numeric field."),
    "specific_regimens": ("Table", "Extract specific regimen names directly from Control regimen text."),
    "tki_agents": ("Table", "Extract TKI agent names directly from Treatment regimen text."),
    "trial_acronym": ("Table", "Extract or normalize the trial acronym directly from Trial name text."),
}


RETRIEVAL_RULES: Dict[str, Tuple[str, str]] = {
    "biomarkers": ("Retrieval", "Biomarker identity is usually not reliably present in Trial name and typically needs external trial/publication evidence."),
    "pdl1_system": ("Retrieval", "PD-L1 scoring system usually requires trial-specific eligibility/publication knowledge beyond Trial name."),
    "pdl1_threshold": ("Retrieval", "PD-L1 threshold usually requires trial-specific eligibility/publication knowledge beyond Trial name."),
    "treatment_setting": ("Retrieval", "Induction vs maintenance usually depends on trial-specific study design, not just the trial name string."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Table/Retrieval labels to expected-key summary CSV.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def decide(expected_key: str) -> Tuple[str, str]:
    if expected_key in TABLE_RULES:
        return TABLE_RULES[expected_key]
    if expected_key in RETRIEVAL_RULES:
        return RETRIEVAL_RULES[expected_key]
    raise KeyError(f"No Table/Retrieval decision configured for expected_key={expected_key!r}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else csv_path

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    add_fields = ["Table or Retrieval", "table_retrieval_rationale"]
    for name in add_fields:
        if name not in fieldnames:
            fieldnames.append(name)

    label_counts: Dict[str, int] = {}
    for row in rows:
        key = (row.get("expected_key") or "").strip()
        label, rationale = decide(key)
        row["Table or Retrieval"] = label
        row["table_retrieval_rationale"] = rationale
        label_counts[label] = label_counts.get(label, 0) + 1

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "input_csv": str(csv_path),
                "output_csv": str(output_path),
                "row_count": len(rows),
                "label_counts": label_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
