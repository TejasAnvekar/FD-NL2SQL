#!/usr/bin/env python3
"""Generate row-level derivation SQLs for table-solvable expected keys."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INVENTORY_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_expected_keys_inventory.csv"
DEFAULT_SUMMARY_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_expected_keys_summary_rettable.csv"
DEFAULT_OUTPUT_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_table_derivation_sqls.csv"
DEFAULT_OUTPUT_JSON = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_table_derivation_sqls.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate derivation SQLs for table-based expected keys.")
    parser.add_argument("--inventory_csv", default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--summary_csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def build_select_variant(sql: str, select_exprs: List[str]) -> str:
    sql0 = (sql or "").strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql0, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not locate FROM clause in SQL: {sql0}")
    return f"SELECT {', '.join(select_exprs)} {sql0[match.start():]};"


def build_base_sql(sql_query: str, column_used: str) -> str:
    select_exprs: List[str] = [quote_ident(col) for col in ("NCT", "PubMed ID", "Trial name")]
    select_exprs.append(f"{quote_ident(column_used)} AS source_value")
    return build_select_variant(sql_query, select_exprs)


def derivation_expression(expected_key: str, column_used: str) -> Tuple[str, str]:
    key = expected_key
    col = column_used

    if key == "class":
        return (
            "derive_ici_class(source_value)",
            "Map generic ICI names to PD-1/PD-L1/CTLA-4.",
        )
    if key == "brand_name":
        return (
            "derive_ici_brand_name(source_value)",
            "Map generic ICI names to brand names.",
        )
    if key == "is_follow_up":
        return (
            "derive_is_follow_up(source_value)",
            "Convert Original publication vs Follow-up text into a boolean-like JSON literal.",
        )
    if key == "is_combination":
        return (
            "derive_is_combination(source_value)",
            "Convert Type of therapy into a boolean-like JSON literal.",
        )
    if key == "followup_type":
        return (
            "derive_followup_type(source_value)",
            "Normalize follow-up type into imaging/clinical/mixed.",
        )
    if key == "combo_type_norm":
        return (
            "derive_combo_type_norm(source_value)",
            "Normalize combination type into canonical labels.",
        )
    if key == "arm_category":
        return (
            "derive_arm_category(source_value)",
            "Bucket number of arms into 1-arm / 2-arm / ≥3-arm.",
        )
    if key == "era":
        return (
            "derive_era_bucket(source_value)",
            "Bucket publication year into ≤2014 / 2015–2018 / ≥2019.",
        )
    if key == "followup_bucket":
        return (
            "derive_followup_bucket(source_value)",
            "Bucket follow-up months into <12 / 12–23 / ≥24.",
        )
    if key == "maturity":
        return (
            "derive_followup_maturity(source_value)",
            "Classify maturity from follow-up months.",
        )
    if key == "phase_bucket":
        return (
            "derive_phase_bucket(source_value)",
            "Bucket trial phase into early vs late.",
        )
    if key == "size_bucket":
        return (
            "derive_size_bucket(source_value)",
            "Bucket sample size into small / medium / large.",
        )
    if key == "setting_normalized":
        return (
            "derive_setting_normalized(source_value)",
            "Normalize clinical setting into canonical categories.",
        )
    if key == "endpoint_role":
        return (
            f"derive_endpoint_role(source_value, {json.dumps(col)})",
            "Infer endpoint role from the source column context.",
        )
    if key == "primary_endpoint_category":
        return (
            "derive_primary_endpoint_category(source_value)",
            "Normalize primary endpoint into canonical labels.",
        )
    if key == "components":
        return (
            "derive_endpoint_components(source_value)",
            "Extract endpoint components as a JSON array string.",
        )
    if key == "irAE_assessed":
        return (
            "derive_irae_assessed(source_value)",
            "Detect whether irAEs were explicitly assessed.",
        )
    if key == "control_category":
        return (
            "derive_control_category(source_value)",
            "Classify the control arm as placebo / active / best supportive care.",
        )
    if key == "control_backbone":
        return (
            "derive_control_backbone(source_value)",
            "Extract chemotherapy backbone from control regimen text.",
        )
    if key == "specific_regimens":
        return (
            "derive_specific_regimens(source_value)",
            "Extract specific regimen names as a JSON array string.",
        )
    if key == "control_drugs":
        return (
            "derive_control_drugs(source_value)",
            "Extract cytotoxic control drugs as a JSON array string.",
        )
    if key == "added_agents":
        return (
            "derive_added_agents(source_value)",
            "Extract non-ICI added agents from the treatment regimen as a JSON array string.",
        )
    if key == "anti_vegf_agents":
        return (
            "derive_anti_vegf_agents(source_value)",
            "Extract anti-VEGF agents from the treatment regimen as a JSON array string.",
        )
    if key == "braf_inhibitor":
        return (
            "derive_braf_inhibitor(source_value)",
            "Extract the BRAF inhibitor from the treatment regimen.",
        )
    if key == "mek_inhibitor":
        return (
            "derive_mek_inhibitor(source_value)",
            "Extract the MEK inhibitor from the treatment regimen.",
        )
    if key == "meki_agents":
        return (
            "derive_meki_agents(source_value)",
            "Extract MEK inhibitors as a JSON array string.",
        )
    if key == "tki_agents":
        return (
            "derive_tki_agents(source_value)",
            "Extract TKI agents as a JSON array string.",
        )
    if key == "schedule":
        return (
            "derive_schedule(source_value)",
            "Extract the dosing schedule from the treatment regimen.",
        )
    if key == "platinum_agent":
        return (
            "derive_platinum_agent(source_value)",
            "Identify cisplatin vs carboplatin from the regimen.",
        )
    if key == "pd_l1_required":
        return (
            "derive_pd_l1_required(source_value)",
            "Convert the PD-L1 requirement flag into a boolean-like JSON literal.",
        )
    if key == "pubmed_url":
        return (
            "derive_pubmed_url(source_value)",
            "Construct a PubMed URL from PubMed ID.",
        )
    if key == "trial_acronym":
        return (
            "derive_trial_acronym(source_value)",
            "Extract or normalize the trial acronym from Trial name.",
        )
    raise KeyError(f"No derivation expression configured for expected_key={expected_key!r}")


def main() -> None:
    args = parse_args()
    inventory_csv = Path(args.inventory_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    with summary_csv.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    key_labels = {
        (row.get("expected_key") or "").strip(): (row.get("Table or Retrieval") or "").strip()
        for row in summary_rows
    }

    with inventory_csv.open(encoding="utf-8", newline="") as handle:
        inventory_rows = list(csv.DictReader(handle))

    out_rows: List[Dict[str, str]] = []
    skipped_rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    for row in inventory_rows:
        key = (row.get("expected_key") or "").strip()
        label = key_labels.get(key, "")
        if label != "Table":
            continue
        column_used = (row.get("column_used") or "").strip()
        sql_query = (row.get("sql_query") or "").strip()
        try:
            base_sql = build_base_sql(sql_query, column_used)
            expr, note = derivation_expression(key, column_used)
            derivation_sql = (
                f"WITH base AS (\n{base_sql.rstrip(';')}\n)\n"
                f"SELECT {quote_ident('NCT')}, {quote_ident('PubMed ID')}, {quote_ident('Trial name')}, "
                f"source_value, {expr} AS {quote_ident(key)}\n"
                f"FROM base;"
            )
            out_row = dict(row)
            out_row["Table or Retrieval"] = label
            out_row["base_sql"] = base_sql
            out_row["derivation_expression"] = expr
            out_row["derivation_notes"] = note
            out_row["derivation_sql"] = derivation_sql
            out_rows.append(out_row)
            counts[key] = counts.get(key, 0) + 1
        except Exception as exc:
            skipped = dict(row)
            skipped["error"] = str(exc)
            skipped_rows.append(skipped)

    fieldnames: List[str] = []
    for row in out_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "inventory_csv": str(inventory_csv),
                "summary_csv": str(summary_csv),
                "output_csv": str(output_csv),
                "row_count": len(out_rows),
                "skipped_count": len(skipped_rows),
                "counts_by_key": dict(sorted(counts.items())),
                "skipped_rows": skipped_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "output_json": str(output_json),
                "row_count": len(out_rows),
                "skipped_count": len(skipped_rows),
                "counts_by_key": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
