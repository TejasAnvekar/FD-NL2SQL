#!/usr/bin/env python3
"""Evaluate augmented-1500 runs, print a summary, and export results to Excel."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
RUNS_ROOT = THIS_DIR / "runs"
FORMATTER = THIS_DIR / "batch_format_llm_predictions_like_ground_truth.py"
TABULAR_EVAL = PROJECT_ROOT / "eval_run_baselines_v3.py"
DERIVED_EVAL = PROJECT_ROOT / "eval_run_baselines_derived.py"

ROOT_KEEP_KEYS = (
    "run_name",
    "manifest_csv",
    "annotated_csv",
    "db_path",
    "model_name",
    "prompt_template_file",
    "max_in_flight",
    "sample_question_count",
    "question_result_count",
    "row_result_count",
    "executor_model_name",
    "executor_api_base",
    "planner_model_name",
    "planner_api_base",
    "cost_currency",
    "planner_input_cost_per_million_tokens",
    "planner_output_cost_per_million_tokens",
    "executor_input_cost_per_million_tokens",
    "executor_output_cost_per_million_tokens",
)

HEADLINE_COLUMNS = (
    "run_name",
    "method",
    "model_name",
    "executor_model_name",
    "planner_model_name",
    "prompt_name",
    "tabular_avg_f1",
    "tabular_exact_table_match_rate",
    "tabular_avg_chrf",
    "tabular_avg_rouge_l_f1",
    "tabular_avg_bertscore_f1",
    "derived_avg_target_row_normalized_f1",
    "derived_avg_target_row_exact_f1",
    "derived_all_rows_target_normalized_rate",
    "derived_all_rows_target_exact_rate",
    "tabular_evaluated_count",
    "derived_evaluated_count",
)

PERCENT_COLUMNS = {
    "tabular_avg_f1",
    "tabular_exact_table_match_rate",
    "tabular_avg_rouge_l_f1",
    "tabular_avg_bertscore_f1",
    "derived_avg_target_row_normalized_f1",
    "derived_avg_target_row_exact_f1",
    "derived_all_rows_target_normalized_rate",
    "derived_all_rows_target_exact_rate",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Discover augmented-1500 run directories, run the table evaluators, "
            "print a terminal summary, and export the aggregated results to Excel."
        )
    )
    ap.add_argument(
        "--run_dirs",
        nargs="*",
        default=[],
        help="Optional explicit run directory paths. If omitted, aug1500 runs are discovered automatically.",
    )
    ap.add_argument(
        "--runs_root",
        default=str(RUNS_ROOT),
        help="Root directory containing run folders for auto-discovery.",
    )
    ap.add_argument(
        "--name_pattern",
        default="aug1500",
        help="Substring used during auto-discovery when no explicit run directories are provided.",
    )
    ap.add_argument(
        "--skip_format",
        type=int,
        default=0,
        help="Skip the formatting step entirely.",
    )
    ap.add_argument(
        "--skip_eval",
        type=int,
        default=0,
        help="Skip the evaluation step entirely and just aggregate existing summaries.",
    )
    ap.add_argument(
        "--force_eval",
        type=int,
        default=0,
        help="Re-run tabular/derived evaluation even if summary files already exist.",
    )
    ap.add_argument(
        "--compute_bertscore",
        type=int,
        default=1,
        help="Pass --compute_bertscore 1 to eval_run_baselines_v3.py.",
    )
    ap.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable to use when launching formatter/evaluators.",
    )
    ap.add_argument(
        "--out_xlsx",
        default=str(RUNS_ROOT / "aug1500_eval_results.xlsx"),
        help="Output Excel workbook path.",
    )
    ap.add_argument(
        "--out_csv",
        default=str(RUNS_ROOT / "aug1500_eval_results.csv"),
        help="Output CSV path for the headline sheet.",
    )
    return ap.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def run_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def discover_run_dirs(runs_root: Path, name_pattern: str) -> list[Path]:
    discovered: list[Path] = []
    for candidate in sorted(runs_root.iterdir()):
        if not candidate.is_dir():
            continue
        summary_path = candidate / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = load_json(summary_path)
        except Exception:
            continue
        manifest_csv = str(summary.get("manifest_csv") or "")
        if "table_question_ground_truths_augmented_1500" in manifest_csv or name_pattern in candidate.name:
            discovered.append(candidate.resolve())
    return discovered


def summary_has_bertscore(tabular_summary_path: Path) -> bool:
    if not tabular_summary_path.exists():
        return False
    try:
        summary = load_json(tabular_summary_path)
    except Exception:
        return False
    value = summary.get("avg_bertscore_f1")
    return isinstance(value, (int, float))


def ensure_evaluated(
    run_dir: Path,
    python_bin: str,
    skip_format: bool,
    skip_eval: bool,
    force_eval: bool,
    compute_bertscore: bool,
) -> None:
    if not skip_format:
        run_cmd([python_bin, str(FORMATTER), "--run_dirs", str(run_dir)])

    if skip_eval:
        return

    tabular_summary_path = run_dir / "tabular_eval_v3" / "summary.json"
    derived_summary_path = run_dir / "derived_eval_v1" / "summary.json"

    need_tabular = force_eval or not tabular_summary_path.exists()
    if compute_bertscore and not summary_has_bertscore(tabular_summary_path):
        need_tabular = True
    if need_tabular:
        cmd = [python_bin, str(TABULAR_EVAL), "--run_dirs", str(run_dir)]
        if compute_bertscore:
            cmd.extend(["--compute_bertscore", "1"])
        run_cmd(cmd)

    need_derived = force_eval or not derived_summary_path.exists()
    if need_derived:
        run_cmd([python_bin, str(DERIVED_EVAL), "--run_dirs", str(run_dir)])


def flatten_token_cost_summary(summary: dict[str, Any]) -> dict[str, Any]:
    token_summary = summary.get("token_cost_summary")
    if not isinstance(token_summary, dict):
        return {}
    flat: dict[str, Any] = {}
    for key, value in token_summary.items():
        if is_scalar(value):
            flat[f"token_cost_{key}"] = value
    return flat


def infer_method(summary: dict[str, Any]) -> str:
    if summary.get("planner_model_name") or summary.get("executor_model_name"):
        return "planner_executor"
    prompt_file = str(summary.get("prompt_template_file") or "").lower()
    run_name = str(summary.get("run_name") or "").lower()
    if "fewshot" in prompt_file or "fewshot" in run_name:
        return "fewshot"
    if "cot" in prompt_file or "cot" in run_name:
        return "cot"
    if "zero_shot" in prompt_file or "zeroshot" in prompt_file or "zeroshot" in run_name:
        return "zeroshot"
    return "unknown"


def prompt_name(summary: dict[str, Any]) -> str:
    prompt_file = summary.get("prompt_template_file")
    if not prompt_file:
        return ""
    return Path(str(prompt_file)).stem


def primary_model_label(summary: dict[str, Any]) -> str:
    if summary.get("model_name"):
        return str(summary["model_name"])
    if summary.get("executor_model_name") and summary.get("planner_model_name"):
        return f'{summary["executor_model_name"]} <-exec | {summary["planner_model_name"]} <-plan'
    return ""


def build_row(run_dir: Path) -> dict[str, Any]:
    root_summary = load_json(run_dir / "summary.json")
    root_summary = dict(root_summary)
    root_summary["run_name"] = run_dir.name

    tabular_summary_path = run_dir / "tabular_eval_v3" / "summary.json"
    derived_summary_path = run_dir / "derived_eval_v1" / "summary.json"

    tabular_summary = load_json(tabular_summary_path) if tabular_summary_path.exists() else {}
    derived_summary = load_json(derived_summary_path) if derived_summary_path.exists() else {}

    row: dict[str, Any] = {}
    for key in ROOT_KEEP_KEYS:
        if key in root_summary and is_scalar(root_summary[key]):
            row[key] = root_summary[key]
    row.update(flatten_token_cost_summary(root_summary))

    row["run_name"] = run_dir.name
    row["method"] = infer_method(root_summary)
    row["model_name"] = primary_model_label(root_summary)
    row["prompt_name"] = prompt_name(root_summary)

    for key, value in tabular_summary.items():
        if is_scalar(value):
            row[f"tabular_{key}"] = value
    for key, value in derived_summary.items():
        if is_scalar(value):
            row[f"derived_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def col_letter(index: int) -> str:
    letters = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def xml_cell(value: Any, cell_ref: str) -> str:
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="inlineStr"><is><t>{str(value).upper()}</t></is></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[list[Any]]) -> str:
    col_widths: dict[int, float] = {}
    for row in rows:
        for idx, value in enumerate(row, start=1):
            width = len("" if value is None else str(value)) + 2
            col_widths[idx] = min(max(col_widths.get(idx, 8), width), 60)

    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width:.2f}" customWidth="1"/>'
        for idx, width in col_widths.items()
    )

    row_xml_parts: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cell_xml = "".join(
            xml_cell(value, f"{col_letter(col_idx)}{row_idx}")
            for col_idx, value in enumerate(row, start=1)
        )
        row_xml_parts.append(f'<row r="{row_idx}">{cell_xml}</row>')

    sheet_data = "".join(row_xml_parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{cols_xml}</cols>"
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    for idx, (sheet_name, _) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{"".join(content_overrides)}'
        "</Types>"
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(workbook_rels)}"
        "</Relationships>"
    )

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles)
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{value * 100:.2f}%"


def format_scalar(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key in PERCENT_COLUMNS and isinstance(value, (int, float)):
        return format_pct(value)
    if key == "tabular_avg_chrf" and isinstance(value, (int, float)):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    columns = list(columns)
    display_rows = [{col: format_scalar(col, row.get(col)) for col in columns} for row in rows]
    widths = {
        col: max(len(col), *(len(display_row.get(col, "")) for display_row in display_rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for display_row in display_rows:
        print(" | ".join(display_row.get(col, "").ljust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()

    if args.run_dirs:
        run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(runs_root, args.name_pattern)

    if not run_dirs:
        raise SystemExit("No aug1500 run directories were found.")

    print(f"Discovered {len(run_dirs)} aug1500 run(s).", flush=True)
    for run_dir in run_dirs:
        print(f" - {run_dir}")

    for run_dir in run_dirs:
        ensure_evaluated(
            run_dir=run_dir,
            python_bin=args.python_bin,
            skip_format=bool(args.skip_format),
            skip_eval=bool(args.skip_eval),
            force_eval=bool(args.force_eval),
            compute_bertscore=bool(args.compute_bertscore),
        )

    rows = [build_row(run_dir) for run_dir in run_dirs]
    rows.sort(key=lambda row: row["run_name"])

    headline_rows = [{col: row.get(col) for col in HEADLINE_COLUMNS} for row in rows]
    all_columns = sorted({key for row in rows for key in row.keys()})

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_xlsx = Path(args.out_xlsx).expanduser().resolve()

    write_csv(out_csv, headline_rows, list(HEADLINE_COLUMNS))

    headline_sheet = [list(HEADLINE_COLUMNS)] + [
        [row.get(col) for col in HEADLINE_COLUMNS] for row in headline_rows
    ]
    full_sheet = [all_columns] + [[row.get(col) for col in all_columns] for row in rows]
    write_xlsx(
        out_xlsx,
        [
            ("headline", headline_sheet),
            ("all_metrics", full_sheet),
        ],
    )

    print()
    print_table(
        headline_rows,
        [
            "run_name",
            "method",
            "model_name",
            "executor_model_name",
            "planner_model_name",
            "tabular_avg_f1",
            "derived_avg_target_row_normalized_f1",
            "derived_avg_target_row_exact_f1",
        ],
    )
    print()
    print(f"Wrote headline CSV: {out_csv}")
    print(f"Wrote Excel workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
