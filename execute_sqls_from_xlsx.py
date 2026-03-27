#!/usr/bin/env python3
"""Extract SQL statements from an XLSX sheet and execute them against SQLite."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"main": MAIN_NS}
SQL_START_RE = re.compile(r"\b(?:SELECT|WITH|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
READ_ONLY_RE = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
PREFERRED_SQL_HEADERS = (
    "sql_query",
    "sql",
    "query_sql",
    "gt_sql",
    "pred_sql",
)


@dataclass
class WorkbookRow:
    excel_row_number: int
    values: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SQL from an XLSX workbook and execute it against a SQLite database.",
    )
    parser.add_argument(
        "--xlsx",
        default="data/cat3_query_sql_llm(2).xlsx",
        help="Path to the XLSX workbook.",
    )
    parser.add_argument(
        "--db",
        default="data/database.db",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Worksheet name to read. Defaults to the first sheet.",
    )
    parser.add_argument(
        "--sql-column",
        default=None,
        help="Header name for the SQL column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path. Defaults to <xlsx-stem>_executed.jsonl next to the workbook.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON summary path. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--max-result-rows",
        type=int,
        default=200,
        help="Maximum number of result rows stored per query in the JSONL output.",
    )
    parser.add_argument(
        "--allow-non-readonly",
        action="store_true",
        help="Allow INSERT/UPDATE/DELETE statements. By default the database is opened read-only.",
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
    letters: List[str] = []
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (header or "").strip().lower()).strip("_")


def load_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("main:si", NS):
        parts = [node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")]
        values.append("".join(parts))
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
    name = chosen.attrib["name"]
    rel_id = chosen.attrib[f"{{{DOC_REL_NS}}}id"]
    rel_target = rel_map[rel_id].lstrip("/")
    if not rel_target.startswith("xl/"):
        rel_target = f"xl/{rel_target}"
    return name, rel_target


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

    if cell_type == "b":
        if value is None or value.text is None:
            return ""
        return "TRUE" if value.text == "1" else "FALSE"

    if value is not None and value.text is not None:
        return value.text

    formula = cell.find("main:f", NS)
    if formula is not None and formula.text is not None:
        return formula.text

    return ""


def build_headers(values_by_index: Dict[int, str]) -> List[Tuple[int, str]]:
    headers: List[Tuple[int, str]] = []
    seen: Dict[str, int] = {}

    for index in sorted(values_by_index):
        raw = values_by_index.get(index, "").strip()
        base = raw or f"column_{column_index_to_letters(index)}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        header = base if count == 0 else f"{base}_{count + 1}"
        headers.append((index, header))

    return headers


def load_sheet_rows(xlsx_path: Path, requested_sheet: Optional[str]) -> Tuple[str, List[str], List[WorkbookRow]]:
    with ZipFile(xlsx_path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_name, sheet_path = resolve_sheet_path(zf, requested_sheet)
        sheet_root = ET.fromstring(zf.read(sheet_path))

    sheet_data = sheet_root.find("main:sheetData", NS)
    if sheet_data is None:
        raise ValueError(f"Sheet {sheet_name!r} does not contain row data.")

    headers: List[Tuple[int, str]] = []
    rows: List[WorkbookRow] = []

    for row in sheet_data.findall("main:row", NS):
        values_by_index: Dict[int, str] = {}
        for cell in row.findall("main:c", NS):
            ref = cell.attrib.get("r", "")
            col_index = column_letters_to_index(ref)
            values_by_index[col_index] = cell_text(cell, shared_strings)

        if not values_by_index or all(not str(v).strip() for v in values_by_index.values()):
            continue

        if not headers:
            headers = build_headers(values_by_index)
            continue

        row_map = {header: values_by_index.get(index, "") for index, header in headers}
        rows.append(
            WorkbookRow(
                excel_row_number=int(row.attrib.get("r", "0") or 0),
                values=row_map,
            )
        )

    if not headers:
        raise ValueError(f"Sheet {sheet_name!r} does not contain a header row.")

    return sheet_name, [header for _, header in headers], rows


def clean_sql(raw_sql: str) -> str:
    text = (raw_sql or "").strip()
    if not text:
        return ""

    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    match = SQL_START_RE.search(text)
    if match:
        text = text[match.start():].strip()

    return text


def looks_like_sql(value: str) -> bool:
    return bool(SQL_START_RE.search(clean_sql(value)))


def detect_sql_column(headers: Sequence[str], rows: Sequence[WorkbookRow], explicit_column: Optional[str]) -> str:
    if explicit_column:
        for header in headers:
            if header == explicit_column:
                return header
        raise ValueError(f"SQL column {explicit_column!r} was not found in headers: {list(headers)}")

    normalized = {header: normalize_header(header) for header in headers}
    for preferred in PREFERRED_SQL_HEADERS:
        for header, normalized_header in normalized.items():
            if normalized_header == preferred:
                return header

    best_header = None
    best_score = -1
    for header in headers:
        sample_values = [row.values.get(header, "") for row in rows[:25]]
        sql_hits = sum(1 for value in sample_values if looks_like_sql(value))
        bonus = 5 if "sql" in normalized[header] else 0
        score = sql_hits + bonus
        if score > best_score:
            best_score = score
            best_header = header

    if best_header and best_score > 0:
        return best_header

    raise ValueError(f"Could not detect a SQL column from headers: {list(headers)}")


def open_sqlite(db_path: Path, allow_non_readonly: bool) -> sqlite3.Connection:
    if allow_non_readonly:
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def result_rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    max_result_rows: int,
    allow_non_readonly: bool,
) -> Dict[str, Any]:
    cleaned = clean_sql(sql)
    if not cleaned:
        return {
            "execution_status": "skipped",
            "error": "Empty SQL cell",
            "result_columns": [],
            "result_row_count": 0,
            "result_rows": [],
            "result_rows_truncated": False,
            "extracted_sql": "",
        }

    if not allow_non_readonly and not READ_ONLY_RE.match(cleaned):
        return {
            "execution_status": "error",
            "error": "Non-read-only SQL is blocked by default. Re-run with --allow-non-readonly if needed.",
            "result_columns": [],
            "result_row_count": 0,
            "result_rows": [],
            "result_rows_truncated": False,
            "extracted_sql": cleaned,
        }

    try:
        cursor = conn.execute(cleaned)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        fetched_rows = cursor.fetchall() if cursor.description else []
        result_rows = result_rows_to_dicts(fetched_rows)
        truncated = max_result_rows >= 0 and len(result_rows) > max_result_rows
        stored_rows = result_rows[:max_result_rows] if truncated else result_rows
        return {
            "execution_status": "ok",
            "error": None,
            "result_columns": columns,
            "result_row_count": len(result_rows),
            "result_rows": stored_rows,
            "result_rows_truncated": truncated,
            "extracted_sql": cleaned,
        }
    except Exception as exc:
        return {
            "execution_status": "error",
            "error": str(exc),
            "result_columns": [],
            "result_row_count": 0,
            "result_rows": [],
            "result_rows_truncated": False,
            "extracted_sql": cleaned,
        }


def default_output_path(xlsx_path: Path) -> Path:
    return xlsx_path.with_name(f"{xlsx_path.stem}_executed.jsonl")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(xlsx_path)
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}.summary.json")
    )

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    sheet_name, headers, workbook_rows = load_sheet_rows(xlsx_path, args.sheet)
    sql_column = detect_sql_column(headers, workbook_rows, args.sql_column)

    conn = open_sqlite(db_path, args.allow_non_readonly)
    output_rows: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    skipped_count = 0

    try:
        for row in workbook_rows:
            execution = execute_sql(
                conn=conn,
                sql=row.values.get(sql_column, ""),
                max_result_rows=args.max_result_rows,
                allow_non_readonly=args.allow_non_readonly,
            )
            status = execution["execution_status"]
            if status == "ok":
                ok_count += 1
            elif status == "error":
                error_count += 1
            else:
                skipped_count += 1

            record = dict(row.values)
            record.update(
                {
                    "excel_row_number": row.excel_row_number,
                    "detected_sql_column": sql_column,
                }
            )
            record.update(execution)
            output_rows.append(record)
    finally:
        conn.close()

    write_jsonl(output_path, output_rows)

    summary = {
        "xlsx_path": str(xlsx_path),
        "db_path": str(db_path),
        "sheet_name": sheet_name,
        "detected_sql_column": sql_column,
        "input_row_count": len(workbook_rows),
        "ok_count": ok_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)

    print(
        f"Processed {len(workbook_rows)} rows from sheet {sheet_name!r}; "
        f"SQL column={sql_column!r}; ok={ok_count}, errors={error_count}, skipped={skipped_count}."
    )
    print(f"JSONL results: {output_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
