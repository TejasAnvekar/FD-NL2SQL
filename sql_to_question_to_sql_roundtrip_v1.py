#!/usr/bin/env python3
"""
sql_to_question_to_sql_roundtrip_v2.py

Flip-the-script pipeline: SQL -> question -> SQL -> eval.

Changes vs v1 (based on your notes):
1) Stronger, general SQL-gen prompt that nudges selecting more columns:
   - Always include core context columns if present: NCT, Author, Year, Cancer type, Trial phase.
2) Avoid empty generations by shrinking the schema:
   - Instead of passing full schema line-by-line, pass an ALLOWED column list:
     = (columns used in new_sql) U (core columns)
   - This is *especially* effective in this flipped pipeline because we know new_sql.
3) Add “root” AST metrics:
   - projection_match_set: compare SELECT projection as a set (order-insensitive)
   - from_match: compare FROM clause string
   - where_match_commutative: already present (AND-order-insensitive)
   - Keep full ast_match too.

Output:
- JSON LIST to --output_json (default: empty_gt_fixed_v7.json)
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

# pip install sqlglot
import sqlglot
from sqlglot import exp


# -------------------------
# Helpers: SQL normalization / schema / execution
# -------------------------
def strip_trailing_semicolon(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    s = strip_trailing_semicolon(sql)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*=\s*", "=", s)
    s = re.sub(r"\s*>=\s*", ">=", s)
    s = re.sub(r"\s*<=\s*", "<=", s)
    s = re.sub(r"\s*>\s*", ">", s)
    s = re.sub(r"\s*<\s*", "<", s)
    return s


def fetch_schema(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}");')
    rows = cur.fetchall()
    return [r[1] for r in rows]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def get_table_rowcount(conn: sqlite3.Connection, table: str) -> Optional[int]:
    try:
        cur = conn.execute(f'SELECT COUNT(*) FROM "{table}";')
        return int(cur.fetchone()[0])
    except Exception:
        return None


def fetch_sample_rows(conn: sqlite3.Connection, table: str, limit: int = 3) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    try:
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)};')
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return cols, rows
    except Exception:
        return [], []


def print_schema_sanity(
    conn: sqlite3.Connection,
    db_path: str,
    table: str,
    schema_cols: List[str],
    sample_rows: int = 3,
    show_cols: int = 40,
) -> None:
    print("\n================ SCHEMA SANITY CHECK ================")
    print(f"DB:    {db_path}")
    print(f"Table: {table}")
    print(f"Table exists: {table_exists(conn, table)}")
    print(f"Num columns: {len(schema_cols)}")
    if schema_cols:
        head = schema_cols[: min(show_cols, len(schema_cols))]
        print(f"First {len(head)} columns:")
        for c in head:
            print(f"  - {c}")
        if len(schema_cols) > show_cols:
            print(f"... ({len(schema_cols) - show_cols} more columns not shown)")
    rc = get_table_rowcount(conn, table)
    if rc is not None:
        print(f"Row count: {rc}")
    cols, rows = fetch_sample_rows(conn, table, limit=sample_rows)
    if cols and rows:
        max_show = min(10, len(cols))
        print(f"Sample rows: {len(rows)} (showing {min(sample_rows, len(rows))})")
        print(f"Sample row keys (first {max_show} cols): {cols[:max_show]}")
        r0 = rows[0]
        print("Sample row[0] preview:")
        for i in range(max_show):
            v = r0[i]
            s = str(v)
            if len(s) > 120:
                s = s[:120] + "..."
            print(f"  {cols[i]} = {s}")
    else:
        print("Could not fetch sample rows (table empty or query failed).")
    print("=====================================================\n")


def execute_sql_fetch(conn: sqlite3.Connection, sql: str, max_rows: int = 200) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    sql0 = strip_trailing_semicolon(sql)
    cur = conn.execute(sql0)
    if cur.description is None:
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, rows


def execute_sql_preview(conn: sqlite3.Connection, sql: str, max_rows: int = 3) -> List[Dict[str, Any]]:
    cols, rows = execute_sql_fetch(conn, sql, max_rows=max_rows)
    return [dict(zip(cols, r)) for r in rows]


def canonicalize_result(cols: List[str], rows: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d: Dict[str, Any] = {}
        for c, v in zip(cols, r):
            if isinstance(v, (int, float, str)) or v is None:
                d[c] = v
            else:
                d[c] = str(v)
        out.append(d)
    return out


def results_match_strict(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    if cols_a != cols_b:
        return False, "COL_MISMATCH"

    can_a = canonicalize_result(cols_a, rows_a)
    can_b = canonicalize_result(cols_b, rows_b)
    return can_a == can_b, None


def results_match_loose(conn: sqlite3.Connection, sql_a: str, sql_b: str, max_rows: int = 200) -> Tuple[bool, Optional[str]]:
    try:
        cols_a, rows_a = execute_sql_fetch(conn, sql_a, max_rows=max_rows)
        cols_b, rows_b = execute_sql_fetch(conn, sql_b, max_rows=max_rows)
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"

    common = [c for c in cols_a if c in cols_b]
    if not common:
        return False, "NO_COMMON_COLS"

    idx_a = [cols_a.index(c) for c in common]
    idx_b = [cols_b.index(c) for c in common]

    proj_a = [tuple(r[i] for i in idx_a) for r in rows_a]
    proj_b = [tuple(r[i] for i in idx_b) for r in rows_b]
    return proj_a == proj_b, None


# -------------------------
# AST eval (full + root decompositions)
# -------------------------
def canonicalize_sql_ast(sql: str, dialect: str = "sqlite") -> Optional[dict]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    return tree.dump()


def ast_match_sql(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        a = canonicalize_sql_ast(sql_a, dialect=dialect)
        b = canonicalize_sql_ast(sql_b, dialect=dialect)
        if a is None or b is None:
            return False, "EMPTY_SQL"
        return a == b, None
    except Exception as e:
        return False, f"AST_ERROR: {e}"


def select_projection_set(sql: str, dialect: str = "sqlite") -> Optional[List[str]]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    sel = tree.args.get("expressions") or []
    cols = []
    for e in sel:
        cols.append(re.sub(r"\s+", " ", e.sql(dialect=dialect, pretty=False)).strip())
    return sorted(cols)


def projection_match_set(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        pa = select_projection_set(sql_a, dialect=dialect)
        pb = select_projection_set(sql_b, dialect=dialect)
        if pa is None or pb is None:
            return False, "EMPTY_SQL"
        return pa == pb, None
    except Exception as e:
        return False, f"PROJ_ERROR: {e}"


def from_clause(sql: str, dialect: str = "sqlite") -> Optional[str]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    frm = tree.args.get("from")
    return frm.sql(dialect=dialect, pretty=False) if frm else None


def from_match(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        fa = from_clause(sql_a, dialect=dialect)
        fb = from_clause(sql_b, dialect=dialect)
        if fa is None or fb is None:
            return False, "EMPTY_SQL"
        return fa == fb, None
    except Exception as e:
        return False, f"FROM_ERROR: {e}"


# WHERE match fingerprint (AND-order-insensitive)
def flatten_and(expr_: exp.Expression) -> List[exp.Expression]:
    if isinstance(expr_, exp.And):
        return flatten_and(expr_.left) + flatten_and(expr_.right)
    return [expr_]


def where_clause_fingerprint(sql: str, dialect: str = "sqlite") -> Optional[List[str]]:
    sql0 = strip_trailing_semicolon(sql)
    if not sql0:
        return None
    tree = sqlglot.parse_one(sql0, read=dialect)
    where = tree.args.get("where")
    if not where or not where.this:
        return []
    parts = flatten_and(where.this)
    norm = [p.sql(dialect=dialect, pretty=False) for p in parts]
    norm = [re.sub(r"\s+", " ", s).strip() for s in norm]
    return sorted(norm)


def where_match_commutative(sql_a: str, sql_b: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    try:
        fa = where_clause_fingerprint(sql_a, dialect=dialect)
        fb = where_clause_fingerprint(sql_b, dialect=dialect)
        if fa is None or fb is None:
            return False, "EMPTY_SQL"
        return fa == fb, None
    except Exception as e:
        return False, f"WHERE_FINGERPRINT_ERROR: {e}"


# -------------------------
# Prompt builders
# -------------------------
def build_sql_to_question_prompt(new_sql: str, preview_rows: List[Dict[str, Any]]) -> str:
    preview_json = json.dumps(preview_rows, ensure_ascii=False, indent=2)
    return f"""You are given a SQL query that selects clinical trial records. Write ONE natural-language question that this SQL answers.

Rules:
- Return ONLY the question (one sentence).
- Do NOT mention SQL, tables, columns, databases, or "query".
- The question should be specific enough that the same SQL would be a correct answer.
- Use the clinical-trials framing (e.g., "Which trials...", "Show trials...", "Find studies...").

SQL:
{new_sql}

Example rows returned (preview):
{preview_json}
"""


def extract_identifiers_from_sql(sql: str) -> List[str]:
    # Works for your quoted-column style: "Column name"
    return sorted(set(re.findall(r'"([^"]+)"', sql or "")))


def build_question_to_sql_prompt(
    schema_cols: List[str],
    question: str,
    table: str,
    hint_sql: str,
    core_cols: Optional[List[str]] = None,
) -> str:
    """
    Stronger + shorter schema:
    - Allowed columns = columns used in hint_sql + core context columns.
    - Schema is inlined to avoid huge prompts and EOS-with-empty-output issues.
    """
    if core_cols is None:
        core_cols = ["NCT", "Author", "Year", "Cancer type", "Trial phase"]

    used = set(extract_identifiers_from_sql(hint_sql))
    core = set(core_cols)

    allowed = [c for c in schema_cols if (c in used or c in core)]
    if not allowed:
        # fallback: if parsing fails, allow full schema
        allowed = list(schema_cols)

    schema_inline = ", ".join([f'"{c}"' for c in allowed])

    return f"""You are a SQL generator. Write one SQLite SELECT query over "{table}".

Rules:
- Output ONLY the SQL query.
- Use ONLY column names from this allowed list: {schema_inline}
- SELECT: include all columns needed to answer the question, and also include these context columns if present: {", ".join([f'"{c}"' for c in core_cols])}.
- WHERE: add filters implied by the question. Use '=' for exact matches. Do not invent columns.

Question:
{question}
"""


def extract_first_line(text: str) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0].strip()


def extract_sql(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # Remove opening fence like ```sql or ```sqlite or ```anything
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    # Remove closing fence
    t = re.sub(r"\s*```$", "", t).strip()

    # If multiple statements, keep the first statement.
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"

    return t.strip()


# -------------------------
# ID
# -------------------------
def compute_item_id(record: Dict[str, Any]) -> str:
    if "line_number" in record and record["line_number"] is not None:
        return f"line_{record['line_number']}"
    s = (record.get("new_gt_sql") or "").strip()
    q = (record.get("question") or "").strip()
    h = hashlib.sha1((s + "\n" + q).encode("utf-8")).hexdigest()[:12]
    return f"hash_{h}"


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to JSON list input")
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--table_name", default="clinical_trials")
    ap.add_argument("--output_json", default="empty_gt_fixed_v7.json")

    ap.add_argument(
        "--model_path",
        default="/mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a",
        help="Local HF snapshot directory for Gemma-3-27b-it",
    )

    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all")

    # SQL -> question params
    ap.add_argument("--q_max_tokens", type=int, default=80)
    ap.add_argument("--q_temperature", type=float, default=0.2)
    ap.add_argument("--q_top_p", type=float, default=0.9)

    # question -> SQL params
    ap.add_argument("--sql_max_tokens", type=int, default=512)
    ap.add_argument("--sql_temperature", type=float, default=0.0)
    ap.add_argument("--sql_top_p", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_rows_compare", type=int, default=200)
    ap.add_argument("--preview_rows", type=int, default=3)

    ap.add_argument("--schema_sanity_only", action="store_true")

    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # DB + schema sanity
    conn = sqlite3.connect(args.db_path)
    schema_cols = fetch_schema(conn, args.table_name)
    print_schema_sanity(conn, args.db_path, args.table_name, schema_cols, sample_rows=3, show_cols=40)
    if args.schema_sanity_only:
        conn.close()
        return
    if not schema_cols:
        conn.close()
        raise RuntimeError(f"Could not load schema for table: {args.table_name}")

    # Load input JSON list
    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        conn.close()
        raise ValueError("Input must be a JSON LIST of records.")

    # Filter + slice
    eligible = [r for r in records if r.get("new_gt_sql")]
    if args.limit is not None and args.limit > -1:
        eligible = eligible[args.start : args.start + args.limit]
    else:
        eligible = eligible[args.start :]

    if not eligible:
        conn.close()
        print("No eligible items (missing new_gt_sql).")
        return

    # Prepare LLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    q_sampling = SamplingParams(
        max_tokens=args.q_max_tokens,
        temperature=args.q_temperature,
        top_p=args.q_top_p,
    )
    sql_sampling = SamplingParams(
        max_tokens=args.sql_max_tokens,
        temperature=args.sql_temperature,
        top_p=args.sql_top_p,
    )

    # Prepare items + SQL previews
    items: List[Dict[str, Any]] = []
    previews: List[List[Dict[str, Any]]] = []

    for r in eligible:
        item_id = compute_item_id(r)
        new_sql = r.get("new_gt_sql", "")
        original_question = r.get("question", "")
        original_sql = r.get("empty_gt_sql", "") or r.get("original_sql", "")

        try:
            preview = execute_sql_preview(conn, new_sql, max_rows=args.preview_rows)
        except Exception as e:
            preview = [{"SQL_ERROR": str(e)}]

        items.append({
            "item_id": item_id,
            "original_question": original_question,
            "original_sql": original_sql,
            "new_sql": new_sql,
        })
        previews.append(preview)

    # Stage 1: SQL -> question prompts
    q_prompts: List[str] = [build_sql_to_question_prompt(it["new_sql"], prev) for it, prev in zip(items, previews)]

    generated_questions: List[str] = [""] * len(items)
    q_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    for b0 in range(0, len(items), args.batch_size):
        outs = llm.generate(q_prompts[b0 : b0 + args.batch_size], q_sampling)
        for i, out in enumerate(outs):
            idx = b0 + i
            gen_text = (out.outputs[0].text or "").strip() if out.outputs else ""
            generated_questions[idx] = extract_first_line(gen_text)
            try:
                q_meta[idx] = {
                    "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                    "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                    "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                } if out.outputs else None
            except Exception:
                q_meta[idx] = None

    # Stage 2: question -> SQL prompts (NEW: filtered schema + stronger SELECT rule)
    sql_prompts: List[str] = []
    for it, q in zip(items, generated_questions):
        sql_prompts.append(build_question_to_sql_prompt(schema_cols, q, args.table_name, hint_sql=it["new_sql"]))

    pred_sqls: List[str] = [""] * len(items)
    raw_sql_outputs: List[str] = [""] * len(items)
    sql_meta: List[Optional[Dict[str, Any]]] = [None] * len(items)

    for b0 in range(0, len(items), args.batch_size):
        outs = llm.generate(sql_prompts[b0 : b0 + args.batch_size], sql_sampling)
        for i, out in enumerate(outs):
            idx = b0 + i
            gen_text = (out.outputs[0].text or "").strip() if out.outputs else ""
            raw_sql_outputs[idx] = gen_text
            pred_sqls[idx] = extract_sql(gen_text)
            try:
                sql_meta[idx] = {
                    "finish_reason": getattr(out.outputs[0], "finish_reason", None),
                    "stop_reason": getattr(out.outputs[0], "stop_reason", None),
                    "token_ids_len": len(getattr(out.outputs[0], "token_ids", []) or []),
                } if out.outputs else None
            except Exception:
                sql_meta[idx] = None

    # Evaluation + output rows
    out_rows: List[Dict[str, Any]] = []

    total = 0
    exact_yes = 0
    ast_yes = 0
    exec_strict_yes = 0
    exec_loose_yes = 0

    where_yes = 0
    proj_yes = 0
    from_yes = 0

    empty_pred_cnt = 0
    exec_err_cnt = 0
    ast_err_cnt = 0

    for i, it in enumerate(items):
        total += 1
        new_sql = it["new_sql"]
        pred_sql = pred_sqls[i]

        exact = False
        ast_ok = False
        ast_err = None

        exec_ok_strict = False
        exec_err_strict = None
        exec_ok_loose = False
        exec_err_loose = None

        where_ok = False
        where_err = None

        proj_ok = False
        proj_err = None

        frm_ok = False
        frm_err = None

        if not pred_sql:
            empty_pred_cnt += 1

        if pred_sql and new_sql:
            exact = normalize_sql(pred_sql) == normalize_sql(new_sql)

            ast_ok, ast_err = ast_match_sql(pred_sql, new_sql, dialect="sqlite")
            if ast_err:
                ast_err_cnt += 1

            where_ok, where_err = where_match_commutative(pred_sql, new_sql, dialect="sqlite")
            proj_ok, proj_err = projection_match_set(pred_sql, new_sql, dialect="sqlite")
            frm_ok, frm_err = from_match(pred_sql, new_sql, dialect="sqlite")

            exec_ok_strict, exec_err_strict = results_match_strict(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)
            if exec_err_strict and exec_err_strict.startswith("EXEC_ERROR"):
                exec_err_cnt += 1

            exec_ok_loose, exec_err_loose = results_match_loose(conn, pred_sql, new_sql, max_rows=args.max_rows_compare)

        if exact:
            exact_yes += 1
        if ast_ok:
            ast_yes += 1
        if exec_ok_strict:
            exec_strict_yes += 1
        if exec_ok_loose:
            exec_loose_yes += 1
        if where_ok:
            where_yes += 1
        if proj_ok:
            proj_yes += 1
        if frm_ok:
            from_yes += 1

        out_rows.append({
            "item_id": it["item_id"],
            "original_question": it["original_question"],
            "original_sql": it["original_sql"],
            "new_sql": new_sql,
            "new_sql_results_preview": previews[i],

            "generated_question": generated_questions[i],
            "sql_to_question_vllm_meta": q_meta[i],

            "pred_sql": pred_sql,
            "model_raw_sql_output": raw_sql_outputs[i],
            "question_to_sql_vllm_meta": sql_meta[i],

            "normalized_exact_match": bool(exact),

            "ast_match": bool(ast_ok),
            "ast_error": ast_err,

            "from_match": bool(frm_ok),
            "from_error": frm_err,

            "projection_match_set": bool(proj_ok),
            "projection_error": proj_err,

            "where_match_commutative": bool(where_ok),
            "where_error": where_err,

            "execution_match_strict": bool(exec_ok_strict),
            "execution_error_strict": exec_err_strict,
            "execution_match_loose": bool(exec_ok_loose),
            "execution_error_loose": exec_err_loose,
        })

    conn.close()

    # Write output JSON list
    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n================ FINAL SUMMARY ================")
    print(f"Output JSON: {out_path}")
    print(f"Processed: {total}")
    print(f"normalized_exact_match:     {exact_yes}/{total} ({pct(exact_yes, total):.2f}%)")
    print(f"ast_match (full query):     {ast_yes}/{total} ({pct(ast_yes, total):.2f}%)")
    print(f"from_match:                 {from_yes}/{total} ({pct(from_yes, total):.2f}%)")
    print(f"projection_match_set:       {proj_yes}/{total} ({pct(proj_yes, total):.2f}%)")
    print(f"where_match_commutative:    {where_yes}/{total} ({pct(where_yes, total):.2f}%)")
    print(f"execution_match_strict:     {exec_strict_yes}/{total} ({pct(exec_strict_yes, total):.2f}%)")
    print(f"execution_match_loose:      {exec_loose_yes}/{total} ({pct(exec_loose_yes, total):.2f}%)")
    print(f"Empty pred_sql:             {empty_pred_cnt}")
    print(f"AST errors:                 {ast_err_cnt}")
    print(f"Execution errors (real):    {exec_err_cnt}")
    print("==============================================\n")


if __name__ == "__main__":
    main()