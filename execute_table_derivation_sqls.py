#!/usr/bin/env python3
"""Execute generated table-derivation SQLs by registering Python UDFs in SQLite."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


DEFAULT_DERIVATION_CSV = "/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_table_derivation_sqls.csv"
DEFAULT_DB = "/mnt/data1/srchowd3/FD-NL2SQL/data/database.db"
DEFAULT_OUTDIR = "/mnt/data1/srchowd3/FD-NL2SQL/data/table_derivation_ground_truths"

ICI_CLASSES = {
    "atezolizumab": "PD-L1",
    "avelumab": "PD-L1",
    "durvalumab": "PD-L1",
    "pembrolizumab": "PD1",
    "nivolumab": "PD1",
    "cemiplimab": "PD1",
    "ipilimumab": "CTLA-4",
    "tremelimumab": "CTLA-4",
}

ICI_BRANDS = {
    "atezolizumab": "Tecentriq",
    "avelumab": "Bavencio",
    "durvalumab": "Imfinzi",
    "pembrolizumab": "Keytruda",
    "nivolumab": "Opdivo",
    "cemiplimab": "Libtayo",
    "ipilimumab": "Yervoy",
    "tremelimumab": "Imjudo",
    "brentuximab-vedotin": "Adcetris",
}

ANTI_VEGF_AGENTS = {
    "bevacizumab",
    "axitinib",
    "cabozantinib",
    "lenvatinib",
    "regorafenib",
    "sunitinib",
}

BRAF_INHIBITORS = {"dabrafenib", "vemurafenib", "encorafenib"}
MEK_INHIBITORS = {"trametinib", "cobimetinib", "binimetinib"}
TKI_AGENTS = {
    "axitinib",
    "cabozantinib",
    "osimertinib",
    "lenvatinib",
    "erlotinib",
    "regorafenib",
    "sunitinib",
}
IGNORE_TOKENS = {
    "placebo",
    "best supportive care",
    "best supportive care ",
    "observation",
    "chemotherapy",
    "investigator's choice chemotherapy",
    "chemoradiotherapy",
    "radiotherapy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute table derivation SQLs and export per-question ground truths.")
    parser.add_argument("--derivation_csv", default=DEFAULT_DERIVATION_CSV)
    parser.add_argument("--db_path", default=DEFAULT_DB)
    parser.add_argument("--output_dir", default=DEFAULT_OUTDIR)
    return parser.parse_args()


def slugify(text: str, max_chars: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "question"
    return slug[:max_chars].rstrip("_") or "question"


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_cell(row.get(key)) for key in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_text(text: Any) -> str:
    text = normalize_space(text)
    text = text.replace("–", "-").replace("—", "-")
    return text


def lower_text(text: Any) -> str:
    return canonical_text(text).lower()


def json_list(values: Iterable[str]) -> Optional[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        val = normalize_space(value).lower()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return json.dumps(out, ensure_ascii=False) if out else None


def split_regimen_tokens(text: Any) -> List[str]:
    raw = canonical_text(text)
    if not raw:
        return []
    raw = re.sub(r"(?i)^\s*(concurrent|phased)\s+", "", raw)
    parts = re.split(r"\s*\+\s*", raw)
    tokens: List[str] = []
    for part in parts:
        subparts = re.split(r"\s*/\s*|\s+or\s+", part, flags=re.IGNORECASE)
        for sub in subparts:
            token = canonical_text(sub).strip(" ,;")
            if token:
                tokens.append(token)
    return tokens


def contains_token(text: Any, token: str) -> bool:
    return token in lower_text(text)


def derive_ici_class(source_value: Any) -> Optional[str]:
    classes: List[str] = []
    seen = set()
    for agent, klass in ICI_CLASSES.items():
        if agent in lower_text(source_value) and klass not in seen:
            classes.append(klass)
            seen.add(klass)
    if not classes:
        return None
    if len(classes) == 1:
        return classes[0]
    return ", ".join(classes)


def derive_ici_brand_name(source_value: Any) -> Optional[str]:
    brands: List[str] = []
    seen = set()
    for agent, brand in ICI_BRANDS.items():
        if agent in lower_text(source_value) and brand not in seen:
            brands.append(brand)
            seen.add(brand)
    if not brands:
        return None
    if len(brands) == 1:
        return brands[0]
    return ", ".join(brands)


def derive_is_follow_up(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    return "true" if "follow-up" in text else "false"


def derive_is_combination(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "combination" in text:
        return "true"
    if "monotherapy" in text:
        return "false"
    return None


def derive_followup_type(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "image" in text or "scan" in text or "radiograph" in text:
        return "imaging"
    if "clinical" in text or "exam" in text:
        return "clinical"
    if "median" in text or "minimum" in text or "upto" in text:
        return None
    return None


def derive_combo_type_norm(source_value: Any) -> Optional[str]:
    text = canonical_text(source_value)
    if not text:
        return None
    if text == "ICI+RadiatICIn":
        return "ICI+Radiation"
    return text


def derive_arm_category(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return None
    try:
        value = int(float(text))
    except Exception:
        return None
    if value <= 1:
        return "1-arm"
    if value == 2:
        return "2-arm"
    return "≥3-arm"


def derive_era_bucket(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return None
    try:
        year = int(float(text))
    except Exception:
        return None
    if year <= 2014:
        return "≤2014"
    if year <= 2018:
        return "2015–2018"
    return "≥2019"


def derive_followup_bucket(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return None
    try:
        months = float(text)
    except Exception:
        return None
    if months < 12:
        return "<12"
    if months < 24:
        return "12–23"
    return "≥24"


def derive_followup_maturity(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return "unknown"
    try:
        months = float(text)
    except Exception:
        return "unknown"
    return "mature" if months >= 24 else "immature"


def derive_phase_bucket(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if any(phase in text for phase in ("phase 3", "phase 4")):
        return "late"
    if any(phase in text for phase in ("phase 1", "phase 1b", "phase 2")):
        return "early"
    return None


def derive_size_bucket(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return None
    try:
        n = int(float(text))
    except Exception:
        return None
    if n < 100:
        return "small"
    if n < 300:
        return "medium"
    return "large"


def derive_setting_normalized(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "neo" in text:
        return "neoadjuvant"
    if "adjuvant" in text:
        return "adjuvant"
    if "periop" in text:
        return "perioperative"
    if "metastatic" in text or "recurrent" in text or "relapsed" in text or "refractory" in text:
        return "metastatic/advanced"
    if "maintenance" in text:
        return "maintenance"
    return canonical_text(source_value)


def derive_endpoint_role(source_value: Any, source_column: Any) -> Optional[str]:
    column = lower_text(source_column)
    if "secondary" in column:
        return "secondary"
    if "primary" in column:
        return "primary"
    return None


def derive_primary_endpoint_category(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "pfs" in text:
        return "PFS"
    if "os" in text:
        return "OS"
    if "orr" in text or "objective response" in text:
        return "ORR"
    if "dfs" in text:
        return "DFS"
    if "dor" in text or "duration of response" in text:
        return "DOR"
    if "safety" in text or "adverse event" in text:
        return "AE-rate"
    return "Other"


def derive_endpoint_components(source_value: Any) -> Optional[str]:
    text = canonical_text(source_value)
    if not text:
        return None
    parts = [canonical_text(part) for part in text.split("+") if canonical_text(part)]
    return json.dumps(parts, ensure_ascii=False) if parts else None


def derive_irae_assessed(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    return "true" if ("irae" in text or "immune-related adverse" in text) else "false"


def derive_control_category(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "placebo" in text:
        return "placebo"
    if "best supportive care" in text or "observation" in text:
        return "best supportive care"
    return "active"


def derive_control_backbone(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if "folfirinox" in text:
        return "FOLFIRINOX"
    if "folfox" in text or "mfolfox6" in text:
        return "FOLFOX"
    if "capox" in text or ("capecitabine" in text and "oxalipl" in text):
        return "CAPOX"
    if "gemcitabine" in text and ("cisplatin" in text or "carboplatin" in text):
        return "Gemcitabine+Platinum"
    if "paclitaxel" in text and "carboplatin" in text:
        return "Carboplatin+Paclitaxel"
    return None


def derive_specific_regimens(source_value: Any) -> Optional[str]:
    backbone = derive_control_backbone(source_value)
    if backbone:
        return json.dumps([backbone], ensure_ascii=False)
    text = lower_text(source_value)
    if "chemotherapy" in text:
        return json.dumps(["Chemotherapy"], ensure_ascii=False)
    return None


def derive_control_drugs(source_value: Any) -> Optional[str]:
    drugs = []
    for token in split_regimen_tokens(source_value):
        low = token.lower()
        if any(ignore in low for ignore in IGNORE_TOKENS):
            continue
        if low in {"platinum agent", "platinum", "platinum-based chemotherapy"}:
            continue
        drugs.append(low)
    return json_list(drugs)


def derive_added_agents(source_value: Any) -> Optional[str]:
    drugs = []
    for token in split_regimen_tokens(source_value):
        low = token.lower()
        if any(agent in low for agent in ICI_CLASSES):
            continue
        if any(ignore in low for ignore in IGNORE_TOKENS):
            continue
        drugs.append(low)
    return json_list(drugs)


def derive_anti_vegf_agents(source_value: Any) -> Optional[str]:
    drugs = [token.lower() for token in split_regimen_tokens(source_value) if token.lower() in ANTI_VEGF_AGENTS]
    return json_list(drugs)


def derive_braf_inhibitor(source_value: Any) -> Optional[str]:
    for token in split_regimen_tokens(source_value):
        low = token.lower()
        if low in BRAF_INHIBITORS:
            return low
    return None


def derive_mek_inhibitor(source_value: Any) -> Optional[str]:
    for token in split_regimen_tokens(source_value):
        low = token.lower()
        if low in MEK_INHIBITORS:
            return low
    return None


def derive_meki_agents(source_value: Any) -> Optional[str]:
    drugs = [token.lower() for token in split_regimen_tokens(source_value) if token.lower() in MEK_INHIBITORS]
    return json_list(drugs)


def derive_tki_agents(source_value: Any) -> Optional[str]:
    drugs = [token.lower() for token in split_regimen_tokens(source_value) if token.lower() in TKI_AGENTS]
    return json_list(drugs)


def derive_schedule(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    match = re.search(r"\b(q\d+w)\b", text)
    if match:
        return match.group(1)
    if "pembrolizumab" in text:
        return "q3w"
    return None


def derive_platinum_agent(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if "carboplatin" in text:
        return "carboplatin"
    if "cisplatin" in text:
        return "cisplatin"
    if "platinum" in text:
        return "platinum agent"
    return None


def derive_pd_l1_required(source_value: Any) -> Optional[str]:
    text = lower_text(source_value)
    if not text:
        return None
    if text.startswith("yes"):
        return "true"
    if text.startswith("no"):
        return "false"
    return None


def derive_pubmed_url(source_value: Any) -> Optional[str]:
    text = normalize_space(source_value)
    if not text:
        return None
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return None
    return f"https://pubmed.ncbi.nlm.nih.gov/{digits}/"


def derive_trial_acronym(source_value: Any) -> Optional[str]:
    text = canonical_text(source_value)
    if not text:
        return None
    patterns = [
        r"KEYNOTE[- ]\d+",
        r"CHECKMATE[- ]?\d+",
        r"IMPOWER ?\d+",
        r"IMPASSION ?\d+",
        r"IMMOTION ?\d+",
        r"IMBLAZE ?\d+",
        r"IMBRAVE ?\d+",
        r"IMSPIRE ?\d+",
        r"IMVIGOR ?\d+",
        r"CA\d{3}-\d+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            token = match.group(0).replace(" ", "-")
            token = re.sub(r"(?i)^impower-", "IMPOWER", token)
            return token.upper() if token.lower().startswith("keynote") or token.lower().startswith("checkmate") else token
    if re.fullmatch(r"[A-Za-z0-9\- ]{2,30}", text):
        return text
    return None


def register_udfs(conn: sqlite3.Connection) -> None:
    funcs: Dict[str, Callable[..., Any]] = {
        "derive_ici_class": derive_ici_class,
        "derive_ici_brand_name": derive_ici_brand_name,
        "derive_is_follow_up": derive_is_follow_up,
        "derive_is_combination": derive_is_combination,
        "derive_followup_type": derive_followup_type,
        "derive_combo_type_norm": derive_combo_type_norm,
        "derive_arm_category": derive_arm_category,
        "derive_era_bucket": derive_era_bucket,
        "derive_followup_bucket": derive_followup_bucket,
        "derive_followup_maturity": derive_followup_maturity,
        "derive_phase_bucket": derive_phase_bucket,
        "derive_size_bucket": derive_size_bucket,
        "derive_setting_normalized": derive_setting_normalized,
        "derive_endpoint_role": derive_endpoint_role,
        "derive_primary_endpoint_category": derive_primary_endpoint_category,
        "derive_endpoint_components": derive_endpoint_components,
        "derive_irae_assessed": derive_irae_assessed,
        "derive_control_category": derive_control_category,
        "derive_control_backbone": derive_control_backbone,
        "derive_specific_regimens": derive_specific_regimens,
        "derive_control_drugs": derive_control_drugs,
        "derive_added_agents": derive_added_agents,
        "derive_anti_vegf_agents": derive_anti_vegf_agents,
        "derive_braf_inhibitor": derive_braf_inhibitor,
        "derive_mek_inhibitor": derive_mek_inhibitor,
        "derive_meki_agents": derive_meki_agents,
        "derive_tki_agents": derive_tki_agents,
        "derive_schedule": derive_schedule,
        "derive_platinum_agent": derive_platinum_agent,
        "derive_pd_l1_required": derive_pd_l1_required,
        "derive_pubmed_url": derive_pubmed_url,
        "derive_trial_acronym": derive_trial_acronym,
    }
    for name, func in funcs.items():
        conn.create_function(name, func.__code__.co_argcount, func)


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text in {"true", "false", "null"} or text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def sanitize_derivation_sql(sql: str) -> str:
    text = (sql or "").strip()
    return re.sub(r";\s*\)\s*SELECT", "\n)\nSELECT", text, count=1, flags=re.IGNORECASE | re.DOTALL)


def main() -> int:
    args = parse_args()
    derivation_csv = Path(args.derivation_csv).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    questions_dir = output_dir / "questions"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    with derivation_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    conn = sqlite3.connect(db_path)
    register_udfs(conn)
    try:
        manifest_rows: List[Dict[str, Any]] = []
        exported_count = 0
        skipped_count = 0
        null_count_by_key: Dict[str, int] = {}
        row_count_by_key: Dict[str, int] = {}

        for row in rows:
            csv_row_number = int(row.get("csv_row_number") or 0)
            item_id = f"row_{csv_row_number}"
            expected_key = (row.get("expected_key") or "").strip()
            question = (row.get("natural_language_query") or "").strip()
            derivation_sql = (row.get("derivation_sql") or "").strip()
            question_dir = questions_dir / f"{slugify(expected_key, 40)}__{item_id}__{slugify(question)}"

            status = "ok"
            error = ""
            result_cols: List[str] = []
            result_rows: List[tuple] = []
            try:
                cur = conn.execute(sanitize_derivation_sql(derivation_sql).rstrip(";"))
                result_cols = [d[0] for d in cur.description] if cur.description else []
                result_rows = cur.fetchall() if cur.description else []

                export_rows: List[Dict[str, Any]] = []
                key_nulls = 0
                value_col = expected_key
                for row_index, values in enumerate(result_rows, start=1):
                    obj = {result_cols[idx]: values[idx] for idx in range(len(result_cols))}
                    parsed_value = parse_jsonish(obj.get(value_col))
                    if parsed_value is None:
                        key_nulls += 1
                    row_out = {"row_index": row_index}
                    for col in result_cols:
                        row_out[col] = obj.get(col)
                    row_out["derived_expected_llm_response"] = json.dumps({expected_key: parsed_value}, ensure_ascii=False)
                    export_rows.append(row_out)

                fieldnames = ["row_index"] + result_cols + ["derived_expected_llm_response"]
                ground_truth_csv = question_dir / "ground_truth_table.csv"
                write_csv(ground_truth_csv, fieldnames, export_rows)
                write_json(
                    question_dir / "metadata.json",
                    {
                        "item_id": item_id,
                        "csv_row_number": csv_row_number,
                        "expected_key": expected_key,
                        "question": question,
                        "column_used": row.get("column_used"),
                        "expected_llm_response": row.get("expected_llm_response"),
                        "derivation_expression": row.get("derivation_expression"),
                        "derivation_notes": row.get("derivation_notes"),
                        "derivation_sql": derivation_sql,
                        "ground_truth_table_csv": str(ground_truth_csv),
                        "row_count": len(export_rows),
                        "null_value_count": key_nulls,
                    },
                )
                exported_count += 1
                row_count_by_key[expected_key] = row_count_by_key.get(expected_key, 0) + len(export_rows)
                null_count_by_key[expected_key] = null_count_by_key.get(expected_key, 0) + key_nulls
            except Exception as exc:
                status = "skipped"
                error = str(exc)
                skipped_count += 1
                write_json(
                    question_dir / "metadata.json",
                    {
                        "item_id": item_id,
                        "csv_row_number": csv_row_number,
                        "expected_key": expected_key,
                        "question": question,
                        "status": status,
                        "error": error,
                        "derivation_sql": derivation_sql,
                    },
                )

            manifest_rows.append(
                {
                    "item_id": item_id,
                    "csv_row_number": csv_row_number,
                    "expected_key": expected_key,
                    "question": question,
                    "status": status,
                    "error": error,
                    "question_dir": str(question_dir),
                    "ground_truth_table_csv": str(question_dir / "ground_truth_table.csv") if status == "ok" else "",
                }
            )
    finally:
        conn.close()

    write_csv(
        manifest_path,
        ["item_id", "csv_row_number", "expected_key", "question", "status", "error", "question_dir", "ground_truth_table_csv"],
        manifest_rows,
    )
    write_json(
        summary_path,
        {
            "derivation_csv": str(derivation_csv),
            "db_path": str(db_path),
            "output_dir": str(output_dir),
            "question_count": len(rows),
            "exported_count": exported_count,
            "skipped_count": skipped_count,
            "row_count_by_key": dict(sorted(row_count_by_key.items())),
            "null_value_count_by_key": dict(sorted(null_count_by_key.items())),
            "manifest_csv": str(manifest_path),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "question_count": len(rows),
                "exported_count": exported_count,
                "skipped_count": skipped_count,
                "manifest_csv": str(manifest_path),
                "summary_json": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
