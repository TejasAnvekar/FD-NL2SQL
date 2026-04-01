#!/usr/bin/env python3
"""Annotate expected_llm_response keys with closest SQLite column matches."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"main": MAIN_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
ANNOTATION_HEADER = "expected_llm_response_key_column_match"
VALUE_ANNOTATION_HEADER = "expected_llm_response_value_column_candidates"
FINAL_COLUMN_HEADER = "final_column"
FINAL_KEY_SCORE_THRESHOLD = 0.75
FINAL_VALUE_SCORE_THRESHOLD = 0.90
STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "by",
    "for",
    "from",
    "given",
    "in",
    "is",
    "of",
    "or",
    "relation",
    "the",
    "to",
}
BOOL_ALIASES = {
    True: ["true", "yes", "1"],
    False: ["false", "no", "0"],
}
REPLACEMENTS = {
    "followup": "follow up",
    "follow_up": "follow up",
    "pd_l1": "pd l1",
    "pdl1": "pd l1",
    "combo": "combination",
    "ici+": "ici ",
}


@dataclass
class MatchResult:
    key: str
    matched_column: Optional[str]
    score: float
    is_match: bool


@dataclass
class SheetRow:
    row_number: int
    values: Dict[str, str]


@dataclass
class DbProfile:
    table_name: str
    columns: List[str]
    values_by_column: Dict[str, List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append a column to the XLSX that shows whether keys inside "
            "expected_llm_response resemble existing SQLite column names."
        ),
    )
    parser.add_argument(
        "--xlsx",
        default="data/cat3_query_sql_llm(2).xlsx",
        help="Input workbook path.",
    )
    parser.add_argument(
        "--db",
        default="data/database.db",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Worksheet name to annotate. Defaults to the first sheet.",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table name to inspect. Defaults to the first SQLite table.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Use .xlsx for a workbook copy or .csv for a flat export.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum similarity score to count as a match.",
    )
    return parser.parse_args()


def column_letters_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = (value * 26) + (ord(ch.upper()) - 64)
    return value


def column_index_to_letters(index: int) -> str:
    if index < 1:
        raise ValueError("Column index must be >= 1")
    chars: List[str] = []
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        chars.append(chr(65 + remainder))
    return "".join(reversed(chars))


def load_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("main:si", NS):
        values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return values


def resolve_sheet_path(zf: ZipFile, requested_sheet: Optional[str]) -> Tuple[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
    }

    sheets = workbook.find("main:sheets", NS)
    if sheets is None or not list(sheets):
        raise ValueError("Workbook does not contain any sheets.")

    chosen = None
    if requested_sheet:
        for sheet in sheets:
            if sheet.attrib.get("name") == requested_sheet:
                chosen = sheet
                break
        if chosen is None:
            available = [sheet.attrib.get("name", "") for sheet in sheets]
            raise ValueError(
                f"Sheet {requested_sheet!r} was not found. Available sheets: {available}"
            )
    else:
        chosen = list(sheets)[0]

    assert chosen is not None
    sheet_name = chosen.attrib["name"]
    rel_id = chosen.attrib[f"{{{DOC_REL_NS}}}id"]
    rel_target = rel_map[rel_id].lstrip("/")
    if not rel_target.startswith("xl/"):
        rel_target = f"xl/{rel_target}"
    return sheet_name, rel_target


def cell_text(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("main:v", NS)

    if cell_type == "s":
        if value is None or value.text is None:
            return ""
        idx = int(value.text)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""

    if cell_type == "inlineStr":
        inline = cell.find("main:is", NS)
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))

    if value is not None and value.text is not None:
        return value.text

    formula = cell.find("main:f", NS)
    if formula is not None and formula.text is not None:
        return formula.text

    return ""


def find_header_column(
    sheet_root: ET.Element,
    shared_strings: Sequence[str],
    header_name: str,
) -> Tuple[int, ET.Element]:
    sheet_data = sheet_root.find("main:sheetData", NS)
    if sheet_data is None:
        raise ValueError("Worksheet does not contain sheetData.")

    header_row = sheet_data.find("main:row", NS)
    if header_row is None:
        raise ValueError("Worksheet does not contain a header row.")

    for cell in header_row.findall("main:c", NS):
        if cell_text(cell, shared_strings).strip() == header_name:
            return column_letters_to_index(cell.attrib.get("r", "")), header_row

    raise ValueError(f"Could not find header {header_name!r} in the worksheet.")


def extract_headers_and_rows(
    sheet_root: ET.Element,
    shared_strings: Sequence[str],
) -> Tuple[List[str], List[SheetRow]]:
    sheet_data = sheet_root.find("main:sheetData", NS)
    if sheet_data is None:
        raise ValueError("Worksheet does not contain sheetData.")

    header_row = sheet_data.find("main:row", NS)
    if header_row is None:
        raise ValueError("Worksheet does not contain a header row.")

    headers_by_index: Dict[int, str] = {}
    ordered_indices: List[int] = []
    duplicate_counts: Dict[str, int] = {}

    for cell in header_row.findall("main:c", NS):
        col_index = column_letters_to_index(cell.attrib.get("r", ""))
        raw_header = cell_text(cell, shared_strings).strip() or f"column_{column_index_to_letters(col_index)}"
        dup_count = duplicate_counts.get(raw_header, 0)
        duplicate_counts[raw_header] = dup_count + 1
        header = raw_header if dup_count == 0 else f"{raw_header}_{dup_count + 1}"
        headers_by_index[col_index] = header
        ordered_indices.append(col_index)

    headers = [headers_by_index[idx] for idx in ordered_indices]
    rows: List[SheetRow] = []
    for row in sheet_data.findall("main:row", NS):
        row_number = int(row.attrib.get("r", "0") or 0)
        if row_number == 1:
            continue

        values_by_index: Dict[int, str] = {}
        for cell in row.findall("main:c", NS):
            col_index = column_letters_to_index(cell.attrib.get("r", ""))
            values_by_index[col_index] = cell_text(cell, shared_strings)

        row_values = {headers_by_index[idx]: values_by_index.get(idx, "") for idx in ordered_indices}
        rows.append(SheetRow(row_number=row_number, values=row_values))

    return headers, rows


def pick_table_name(conn: sqlite3.Connection, requested_table: Optional[str]) -> str:
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]
    if not tables:
        raise ValueError("No SQLite tables found.")
    if requested_table:
        if requested_table not in tables:
            raise ValueError(f"Table {requested_table!r} was not found. Available tables: {tables}")
        return requested_table
    return tables[0]


def fetch_db_columns(db_path: Path, requested_table: Optional[str]) -> Tuple[str, List[str]]:
    conn = sqlite3.connect(db_path)
    try:
        table_name = pick_table_name(conn, requested_table)
        cols = [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')]
        return table_name, cols
    finally:
        conn.close()


def fetch_db_profile(db_path: Path, requested_table: Optional[str]) -> DbProfile:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        table_name = pick_table_name(conn, requested_table)
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')]
        rows = conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
        values_by_column: Dict[str, List[str]] = {}
        for column in columns:
            seen: Set[str] = set()
            values: List[str] = []
            for row in rows:
                value = row[column]
                if value is None:
                    continue
                text = str(value).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                values.append(text)
            values_by_column[column] = values
        return DbProfile(
            table_name=table_name,
            columns=columns,
            values_by_column=values_by_column,
        )
    finally:
        conn.close()


def normalize_tokens(text: str) -> Tuple[str, Set[str]]:
    normalized = text.lower()
    for old, new in REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"(?<=[a-z])(?=[0-9])", " ", normalized)
    normalized = re.sub(r"(?<=[0-9])(?=[a-z])", " ", normalized)
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    raw_tokens = [token for token in normalized.split() if token]

    tokens: List[str] = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)

    return " ".join(tokens), set(tokens)


def score_column_match(key: str, column_name: str) -> float:
    key_phrase, key_tokens = normalize_tokens(key)
    col_phrase, col_tokens = normalize_tokens(column_name)
    if not key_phrase or not col_phrase:
        return 0.0

    overlap = len(key_tokens & col_tokens) / len(key_tokens) if key_tokens else 0.0
    jaccard = len(key_tokens & col_tokens) / len(key_tokens | col_tokens) if (key_tokens | col_tokens) else 0.0
    seq_ratio = SequenceMatcher(None, key_phrase, col_phrase).ratio()
    contains_bonus = 0.20 if key_phrase in col_phrase or col_phrase in key_phrase else 0.0
    subset_bonus = 0.25 if key_tokens and key_tokens.issubset(col_tokens) else 0.0
    return min(1.0, max(overlap, jaccard, seq_ratio) + contains_bonus + subset_bonus)


def best_column_match(key: str, db_columns: Sequence[str], threshold: float) -> MatchResult:
    scored = [(score_column_match(key, column), column) for column in db_columns]
    score, column_name = max(scored, key=lambda item: item[0])
    return MatchResult(
        key=key,
        matched_column=column_name,
        score=score,
        is_match=score >= threshold,
    )


def scalar_to_search_terms(value: Any) -> List[str]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return BOOL_ALIASES[value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
        if isinstance(value, float) and value.is_integer():
            return [str(int(value)), text]
        return [text]
    text = str(value).strip()
    return [text] if text else []


def flatten_value_terms(value: Any) -> List[str]:
    if isinstance(value, list):
        terms: List[str] = []
        for item in value:
            terms.extend(flatten_value_terms(item))
        return terms
    return scalar_to_search_terms(value)


def parse_number(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def score_value_against_cell(search_term: str, cell_value: str) -> float:
    raw_search = (search_term or "").strip()
    raw_cell = (cell_value or "").strip()
    if not raw_search or not raw_cell:
        return 0.0

    if raw_search.lower() == raw_cell.lower():
        return 1.0

    search_num = parse_number(raw_search)
    cell_num = parse_number(raw_cell)
    if search_num is not None and cell_num is not None and search_num == cell_num:
        return 1.0

    search_phrase, search_tokens = normalize_tokens(raw_search)
    cell_phrase, cell_tokens = normalize_tokens(raw_cell)
    if not search_phrase or not cell_phrase:
        return 0.0

    if search_phrase == cell_phrase:
        return 1.0

    overlap = len(search_tokens & cell_tokens) / len(search_tokens) if search_tokens else 0.0
    jaccard = len(search_tokens & cell_tokens) / len(search_tokens | cell_tokens) if (search_tokens | cell_tokens) else 0.0
    seq_ratio = SequenceMatcher(None, search_phrase, cell_phrase).ratio()
    contains_bonus = 0.18 if search_phrase in cell_phrase or cell_phrase in search_phrase else 0.0
    subset_bonus = 0.22 if search_tokens and search_tokens.issubset(cell_tokens) else 0.0
    return min(1.0, max(overlap, jaccard, seq_ratio) + contains_bonus + subset_bonus)


def find_value_candidate_columns(
    value: Any,
    db_profile: DbProfile,
    threshold: float,
    top_k: int = 5,
) -> Tuple[List[Tuple[str, float]], Optional[Tuple[str, float]]]:
    search_terms = flatten_value_terms(value)
    if not search_terms:
        return [], None

    scored_columns: List[Tuple[str, float]] = []
    for column in db_profile.columns:
        column_values = db_profile.values_by_column.get(column, [])
        if not column_values:
            continue

        best_scores: List[float] = []
        for term in search_terms:
            best = 0.0
            for cell_value in column_values:
                candidate_score = score_value_against_cell(term, cell_value)
                if candidate_score > best:
                    best = candidate_score
                    if best >= 0.999:
                        break
            best_scores.append(best)

        if not best_scores:
            continue

        coverage = sum(best_scores) / len(best_scores)
        exactish_hits = sum(1 for score in best_scores if score >= 0.95)
        aggregate = min(1.0, coverage + (0.05 * exactish_hits))
        scored_columns.append((column, aggregate))

    if not scored_columns:
        return [], None

    scored_columns.sort(key=lambda item: (-item[1], item[0]))
    best_overall = scored_columns[0]
    matches = [item for item in scored_columns if item[1] >= threshold][:top_k]
    return matches, best_overall


def format_json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def annotation_from_payload(
    payload_text: str,
    db_columns: Sequence[str],
    threshold: float,
) -> str:
    payload_text = (payload_text or "").strip()
    if not payload_text:
        return "empty expected_llm_response"

    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        return f"invalid JSON: {exc}"

    if not isinstance(payload, dict):
        return f"JSON is {type(payload).__name__}, expected an object"

    parts: List[str] = []
    for key in payload:
        match = best_column_match(key, db_columns, threshold)
        if match.is_match:
            parts.append(
                f"{key} -> {match.matched_column} [MATCH {match.score:.2f}]"
            )
        else:
            parts.append(
                f"{key} -> no close match (best: {match.matched_column}, {match.score:.2f})"
            )
    return " | ".join(parts)


def value_annotation_from_payload(
    payload_text: str,
    db_profile: DbProfile,
    threshold: float,
) -> str:
    payload_text = (payload_text or "").strip()
    if not payload_text:
        return "empty expected_llm_response"

    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        return f"invalid JSON: {exc}"

    if not isinstance(payload, dict):
        return f"JSON is {type(payload).__name__}, expected an object"

    parts: List[str] = []
    for key, value in payload.items():
        matches, best = find_value_candidate_columns(value, db_profile, threshold)
        if matches:
            rendered = ", ".join(f"{column} [{score:.2f}]" for column, score in matches)
            parts.append(f"{key}={format_json_value(value)} -> {rendered}")
        elif best is not None:
            parts.append(
                f"{key}={format_json_value(value)} -> no close value match "
                f"(best: {best[0]}, {best[1]:.2f})"
            )
        else:
            parts.append(f"{key}={format_json_value(value)} -> no value candidates")
    return " | ".join(parts)


def final_column_from_payload(payload_text: str, db_profile: DbProfile) -> str:
    payload_text = (payload_text or "").strip()
    if not payload_text:
        return "no_match"

    try:
        payload = json.loads(payload_text)
    except Exception:
        return "no_match"

    if not isinstance(payload, dict):
        return "no_match"

    chosen: List[str] = []
    for key, value in payload.items():
        key_match = best_column_match(key, db_profile.columns, threshold=0.0)
        if key_match.matched_column and key_match.score > FINAL_KEY_SCORE_THRESHOLD:
            selected = key_match.matched_column
        else:
            _, best_value_match = find_value_candidate_columns(
                value=value,
                db_profile=db_profile,
                threshold=FINAL_VALUE_SCORE_THRESHOLD,
                top_k=1,
            )
            if best_value_match is not None and best_value_match[1] > FINAL_VALUE_SCORE_THRESHOLD:
                selected = best_value_match[0]
            else:
                selected = "no_match"

        chosen.append(f"{key} -> {selected}")

    if len(chosen) == 1:
        return chosen[0].split(" -> ", 1)[1]
    return " | ".join(chosen)


def make_inline_string_cell(ref: str, text: str, style_id: Optional[str]) -> ET.Element:
    attrs = {"r": ref, "t": "inlineStr"}
    if style_id is not None:
        attrs["s"] = style_id
    cell = ET.Element(f"{{{MAIN_NS}}}c", attrs)
    inline = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
    text_node = ET.SubElement(inline, f"{{{MAIN_NS}}}t")
    if text.startswith(" ") or text.endswith(" ") or "\n" in text:
        text_node.set(XML_SPACE, "preserve")
    text_node.text = text
    return cell


def update_dimension(sheet_root: ET.Element, new_col_index: int) -> None:
    dimension = sheet_root.find("main:dimension", NS)
    if dimension is None:
        return

    ref = dimension.attrib.get("ref", "")
    if not ref:
        return

    if ":" in ref:
        start_ref, end_ref = ref.split(":", 1)
    else:
        start_ref = end_ref = ref

    match = re.match(r"([A-Z]+)(\d+)", end_ref)
    if not match:
        return

    new_end = f"{column_index_to_letters(new_col_index)}{match.group(2)}"
    if start_ref == end_ref:
        dimension.set("ref", f"{start_ref}:{new_end}")
    else:
        dimension.set("ref", f"{start_ref}:{new_end}")


def set_row_spans(row: ET.Element, new_col_index: int) -> None:
    if "spans" in row.attrib:
        row.attrib["spans"] = f"1:{new_col_index}"


def copy_workbook_with_annotation(
    input_path: Path,
    output_path: Path,
    sheet_path: str,
    updated_sheet_bytes: bytes,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(input_path, "r") as src, ZipFile(output_path, "w", compression=ZIP_DEFLATED) as dst:
        for info in src.infolist():
            payload = updated_sheet_bytes if info.filename == sheet_path else src.read(info.filename)
            dst.writestr(info, payload)


def write_csv_annotation(
    xlsx_path: Path,
    db_profile: DbProfile,
    sheet_name: Optional[str],
    output_path: Path,
    threshold: float,
) -> Tuple[str, int]:
    with ZipFile(xlsx_path, "r") as zf:
        shared_strings = load_shared_strings(zf)
        resolved_sheet_name, sheet_path = resolve_sheet_path(zf, sheet_name)
        sheet_root = ET.fromstring(zf.read(sheet_path))

    headers, rows = extract_headers_and_rows(sheet_root, shared_strings)
    if "expected_llm_response" not in headers:
        raise ValueError("Could not find 'expected_llm_response' in worksheet headers.")

    output_headers = list(headers) + [ANNOTATION_HEADER, VALUE_ANNOTATION_HEADER, FINAL_COLUMN_HEADER]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers)
        writer.writeheader()
        for row in rows:
            key_annotation_text = annotation_from_payload(
                row.values.get("expected_llm_response", ""),
                db_profile.columns,
                threshold,
            )
            value_annotation_text = value_annotation_from_payload(
                row.values.get("expected_llm_response", ""),
                db_profile,
                threshold,
            )
            final_column_text = final_column_from_payload(
                row.values.get("expected_llm_response", ""),
                db_profile,
            )
            out_row = dict(row.values)
            out_row[ANNOTATION_HEADER] = key_annotation_text
            out_row[VALUE_ANNOTATION_HEADER] = value_annotation_text
            out_row[FINAL_COLUMN_HEADER] = final_column_text
            writer.writerow(out_row)

    return resolved_sheet_name, len(rows)


def annotate_workbook(
    xlsx_path: Path,
    db_profile: DbProfile,
    sheet_name: Optional[str],
    output_path: Path,
    threshold: float,
) -> Tuple[str, int]:
    with ZipFile(xlsx_path, "r") as zf:
        shared_strings = load_shared_strings(zf)
        resolved_sheet_name, sheet_path = resolve_sheet_path(zf, sheet_name)
        sheet_root = ET.fromstring(zf.read(sheet_path))

    target_col_index, header_row = find_header_column(
        sheet_root=sheet_root,
        shared_strings=shared_strings,
        header_name="expected_llm_response",
    )
    annotation_col_index = target_col_index + 1
    value_annotation_col_index = target_col_index + 2
    final_annotation_col_index = target_col_index + 3

    sheet_data = sheet_root.find("main:sheetData", NS)
    if sheet_data is None:
        raise ValueError("Worksheet does not contain sheetData.")

    header_style = None
    target_col_letter = column_index_to_letters(target_col_index)
    annotation_col_letter = column_index_to_letters(annotation_col_index)
    value_annotation_col_letter = column_index_to_letters(value_annotation_col_index)
    final_annotation_col_letter = column_index_to_letters(final_annotation_col_index)
    for cell in header_row.findall("main:c", NS):
        if cell.attrib.get("r") == f"{target_col_letter}1":
            header_style = cell.attrib.get("s")
            break

    header_row.append(
        make_inline_string_cell(
            ref=f"{annotation_col_letter}1",
            text=ANNOTATION_HEADER,
            style_id=header_style,
        )
    )
    header_row.append(
        make_inline_string_cell(
            ref=f"{value_annotation_col_letter}1",
            text=VALUE_ANNOTATION_HEADER,
            style_id=header_style,
        )
    )
    header_row.append(
        make_inline_string_cell(
            ref=f"{final_annotation_col_letter}1",
            text=FINAL_COLUMN_HEADER,
            style_id=header_style,
        )
    )
    set_row_spans(header_row, final_annotation_col_index)

    annotated_rows = 0
    for row in sheet_data.findall("main:row", NS):
        row_number = int(row.attrib.get("r", "0") or 0)
        if row_number == 1:
            continue

        target_cell = None
        for cell in row.findall("main:c", NS):
            if cell.attrib.get("r") == f"{target_col_letter}{row_number}":
                target_cell = cell
                break

        style_id = target_cell.attrib.get("s") if target_cell is not None else None
        payload_text = cell_text(target_cell, shared_strings) if target_cell is not None else ""
        key_annotation_text = annotation_from_payload(payload_text, db_profile.columns, threshold)
        value_annotation_text = value_annotation_from_payload(payload_text, db_profile, threshold)
        final_column_text = final_column_from_payload(payload_text, db_profile)
        row.append(
            make_inline_string_cell(
                ref=f"{annotation_col_letter}{row_number}",
                text=key_annotation_text,
                style_id=style_id,
            )
        )
        row.append(
            make_inline_string_cell(
                ref=f"{value_annotation_col_letter}{row_number}",
                text=value_annotation_text,
                style_id=style_id,
            )
        )
        row.append(
            make_inline_string_cell(
                ref=f"{final_annotation_col_letter}{row_number}",
                text=final_column_text,
                style_id=style_id,
            )
        )
        set_row_spans(row, final_annotation_col_index)
        annotated_rows += 1

    update_dimension(sheet_root, final_annotation_col_index)
    updated_sheet_bytes = ET.tostring(sheet_root, encoding="utf-8", xml_declaration=True)
    copy_workbook_with_annotation(
        input_path=xlsx_path,
        output_path=output_path,
        sheet_path=sheet_path,
        updated_sheet_bytes=updated_sheet_bytes,
    )
    return resolved_sheet_name, annotated_rows


def default_output_path(xlsx_path: Path) -> Path:
    return xlsx_path.with_name(f"{xlsx_path.stem}_with_key_matches.xlsx")


def main() -> int:
    args = parse_args()
    xlsx_path = Path(args.xlsx).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(xlsx_path)

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    db_profile = fetch_db_profile(db_path, args.table)
    if output_path.suffix.lower() == ".csv":
        resolved_sheet_name, annotated_rows = write_csv_annotation(
            xlsx_path=xlsx_path,
            db_profile=db_profile,
            sheet_name=args.sheet,
            output_path=output_path,
            threshold=args.threshold,
        )
    else:
        resolved_sheet_name, annotated_rows = annotate_workbook(
            xlsx_path=xlsx_path,
            db_profile=db_profile,
            sheet_name=args.sheet,
            output_path=output_path,
            threshold=args.threshold,
        )

    print(
        f"Annotated {annotated_rows} rows in sheet {resolved_sheet_name!r} "
        f"using SQLite table {db_profile.table_name!r}."
    )
    print(f"Output workbook: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
