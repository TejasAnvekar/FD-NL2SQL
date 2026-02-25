#!/usr/bin/env python3
"""Hybrid retrieval over NL->SQL seed candidates.

This script ranks candidate (question, sql) pairs against an input natural-language
question using a weighted score with these components:
- lexical similarity (token cosine + token jaccard)
- character similarity (SequenceMatcher)
- SQL literal coverage in the question
- operator cue consistency
- column mention overlap

Sources:
- seed JSON (default: data/seed_questions.json with decomposed_query blocks)
- generic candidate JSON list
- sqlite table containing question/sql columns
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")
LITERAL_RE = re.compile(r"'((?:''|[^'])*)'")
QUOTED_IDENT_RE = re.compile(r'"([^"]+)"')

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
    "where",
    "were",
    "when",
    "who",
    "whom",
    "whose",
}

# Weight defaults; override from CLI if needed.
W_LEXICAL = 0.40
W_CHAR = 0.20
W_LITERAL = 0.25
W_OPERATOR = 0.10
W_COLUMN = 0.05


@dataclass
class Candidate:
    candidate_id: str
    question: str
    sql: str
    parent_question: Optional[str] = None
    source: str = "seed_json"


@dataclass
class MatchResult:
    rank: int
    total_score: float
    lexical_score: float
    char_score: float
    literal_score: float
    operator_score: float
    column_score: float
    candidate: Candidate


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_counter(tokens: Iterable[str]) -> Counter:
    return Counter(tokens)


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for tok, val in a.items():
        dot += val * b.get(tok, 0)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_set(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def extract_sql_literals(sql: str) -> List[str]:
    vals: List[str] = []
    for m in LITERAL_RE.findall(sql or ""):
        vals.append(m.replace("''", "'").strip())
    return [v for v in vals if v]


def extract_sql_columns(sql: str) -> List[str]:
    cols = [c.strip() for c in QUOTED_IDENT_RE.findall(sql or "")]
    # Keep order but de-duplicate.
    out: List[str] = []
    seen = set()
    for c in cols:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def phrase_coverage(phrase: str, q_tokens: Sequence[str], q_norm: str) -> float:
    p_norm = normalize_text(phrase)
    if not p_norm:
        return 0.0
    if p_norm in q_norm:
        return 1.0
    p_toks = tokenize(phrase)
    if not p_toks:
        return 0.0
    p_set = set(p_toks)
    q_set = set(q_tokens)
    return len(p_set & q_set) / max(1, len(p_set))


def literal_coverage_score(sql: str, q_tokens: Sequence[str], q_norm: str) -> float:
    literals = extract_sql_literals(sql)
    if not literals:
        # Neutral-low score when there are no literal constraints.
        return 0.30
    cov = [phrase_coverage(v, q_tokens, q_norm) for v in literals]
    return float(sum(cov) / len(cov)) if cov else 0.0


def column_mention_score(sql: str, q_tokens: Sequence[str]) -> float:
    cols = extract_sql_columns(sql)
    if not cols:
        return 0.0
    q_set = set(q_tokens)
    scores: List[float] = []
    for col in cols:
        c_toks = [t for t in tokenize(col) if t not in STOPWORDS]
        if not c_toks:
            continue
        c_set = set(c_toks)
        scores.append(len(c_set & q_set) / len(c_set))
    if not scores:
        return 0.0
    # Favor best overlapping field mention, without heavily penalizing wide SELECT lists.
    return max(scores)


def sql_operator_set(sql: str) -> set[str]:
    sql_l = (sql or "").lower()
    raw_ops = re.findall(r"(>=|<=|<>|!=|=|>|<|\blike\b)", sql_l)
    mapped = set()
    for op in raw_ops:
        if op == ">=":
            mapped.add("ge")
        elif op == "<=":
            mapped.add("le")
        elif op in {"!=", "<>"}:
            mapped.add("neq")
        elif op == ">":
            mapped.add("gt")
        elif op == "<":
            mapped.add("lt")
        elif op == "like":
            mapped.add("like")
        elif op == "=":
            mapped.add("eq")
    return mapped


def question_operator_set(question: str) -> set[str]:
    q = (question or "").lower()
    ops = set()

    if re.search(r"\b(at least|no less than|minimum of|or more)\b", q):
        ops.add("ge")
    if re.search(r"\b(at most|no more than|up to|maximum of)\b", q):
        ops.add("le")
    if re.search(r"\b(greater than|more than|above|over)\b", q):
        ops.add("gt")
    if re.search(r"\b(less than|below|under)\b", q):
        ops.add("lt")
    if re.search(r"\b(not equal|different from|except)\b", q):
        ops.add("neq")
    if re.search(r"\b(contains|containing|starts with|ends with|like)\b", q):
        ops.add("like")
    if re.search(r"\b(equal to|equals|exactly)\b", q):
        ops.add("eq")

    return ops


def operator_match_score(question: str, sql: str) -> float:
    q_ops = question_operator_set(question)
    s_ops = sql_operator_set(sql)

    if not q_ops and not s_ops:
        return 1.0
    if not q_ops:
        return 0.50
    if not s_ops:
        return 0.0

    overlap = len(q_ops & s_ops)
    p = overlap / len(s_ops)
    r = overlap / len(q_ops)
    return (2.0 * p * r / (p + r)) if (p + r) else 0.0


def score_candidate(question: str, candidate: Candidate) -> Dict[str, float]:
    q_toks = tokenize(question)
    c_toks = tokenize(candidate.question)

    q_counter = token_counter(q_toks)
    c_counter = token_counter(c_toks)

    token_cos = cosine_counter(q_counter, c_counter)
    token_jac = jaccard_set(q_toks, c_toks)
    lexical = 0.70 * token_cos + 0.30 * token_jac

    char_score = sequence_ratio(question, candidate.question)
    q_norm = normalize_text(question)
    literal_score = literal_coverage_score(candidate.sql, q_toks, q_norm)
    op_score = operator_match_score(question, candidate.sql)
    col_score = column_mention_score(candidate.sql, q_toks)

    total = (
        W_LEXICAL * lexical
        + W_CHAR * char_score
        + W_LITERAL * literal_score
        + W_OPERATOR * op_score
        + W_COLUMN * col_score
    )

    return {
        "total": total,
        "lexical": lexical,
        "char": char_score,
        "literal": literal_score,
        "operator": op_score,
        "column": col_score,
    }


def rank_candidates(question: str, candidates: Sequence[Candidate], top_k: int) -> List[MatchResult]:
    scored = []
    for cand in candidates:
        s = score_candidate(question, cand)
        scored.append((s, cand))

    scored.sort(key=lambda x: x[0]["total"], reverse=True)

    results: List[MatchResult] = []
    for idx, (s, cand) in enumerate(scored[: max(1, top_k)], start=1):
        results.append(
            MatchResult(
                rank=idx,
                total_score=float(s["total"]),
                lexical_score=float(s["lexical"]),
                char_score=float(s["char"]),
                literal_score=float(s["literal"]),
                operator_score=float(s["operator"]),
                column_score=float(s["column"]),
                candidate=cand,
            )
        )
    return results


def load_candidates_from_seed_json(path: Path) -> List[Candidate]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[Candidate] = []

    def add_candidate(cand_id: str, q: str, sql: str, parent: Optional[str], source: str) -> None:
        q = (q or "").strip()
        sql = (sql or "").strip()
        if not q or not sql:
            return
        out.append(
            Candidate(
                candidate_id=cand_id,
                question=q,
                sql=sql,
                parent_question=parent,
                source=source,
            )
        )

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            parent = item.get("original_question") or item.get("question") or item.get("natural_question")

            dq = item.get("decomposed_query")
            if isinstance(dq, dict):
                for q_key, payload in dq.items():
                    if not isinstance(payload, dict):
                        continue
                    add_candidate(
                        cand_id=f"seed[{i}].{q_key}",
                        q=payload.get("question", ""),
                        sql=payload.get("sql", ""),
                        parent=parent,
                        source="seed_json/decomposed",
                    )
                continue

            # Fallback for flat question/sql rows.
            add_candidate(
                cand_id=f"seed[{i}]",
                q=item.get("question", ""),
                sql=item.get("sql") or item.get("gt_sql") or "",
                parent=parent,
                source="seed_json/flat",
            )

    elif isinstance(obj, dict):
        for q_key, payload in obj.items():
            if not isinstance(payload, dict):
                continue
            add_candidate(
                cand_id=f"seed.{q_key}",
                q=payload.get("question", ""),
                sql=payload.get("sql", ""),
                parent=None,
                source="seed_json/dict",
            )

    return out


def quote_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def load_candidates_from_sqlite(
    db_path: Path,
    table: str,
    question_col: str,
    sql_col: str,
    id_col: Optional[str],
) -> List[Candidate]:
    out: List[Candidate] = []
    conn = sqlite3.connect(str(db_path))
    try:
        if id_col:
            q = (
                f"SELECT {quote_ident(id_col)}, {quote_ident(question_col)}, {quote_ident(sql_col)} "
                f"FROM {quote_ident(table)}"
            )
        else:
            q = (
                f"SELECT rowid AS _rowid_, {quote_ident(question_col)}, {quote_ident(sql_col)} "
                f"FROM {quote_ident(table)}"
            )
        rows = conn.execute(q).fetchall()
        for row in rows:
            cand_id = str(row[0])
            question = str(row[1] or "")
            sql = str(row[2] or "")
            if question.strip() and sql.strip():
                out.append(
                    Candidate(
                        candidate_id=cand_id,
                        question=question,
                        sql=sql,
                        source=f"sqlite:{table}",
                    )
                )
    finally:
        conn.close()
    return out


def load_question_from_dataset(path: Path, index: int, key: str) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if index < 0 or index >= len(obj):
        raise IndexError(f"question-index {index} is out of range for {len(obj)} rows")

    row = obj[index]
    if not isinstance(row, dict):
        raise ValueError(f"Row {index} is not an object")

    q = row.get(key)
    if not isinstance(q, str) or not q.strip():
        # Common fallbacks.
        for alt in ("natural_question", "question", "original_question"):
            if isinstance(row.get(alt), str) and row[alt].strip():
                return row[alt].strip()
        raise KeyError(f"Could not find non-empty question text using key={key!r}")
    return q.strip()


def is_safe_readonly_sql(sql: str) -> Tuple[bool, str]:
    s = (sql or "").strip()
    if not s:
        return False, "empty"
    if ";" in s.rstrip(";"):
        return False, "multiple_statements"

    head = s.lstrip().split(None, 1)[0].lower()
    if head not in {"select", "with"}:
        return False, f"not_select_or_with:{head}"

    banned = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "replace",
    }
    toks = set(re.findall(r"[a-z_]+", s.lower()))
    bad = sorted(toks & banned)
    if bad:
        return False, f"contains_banned:{bad}"
    return True, ""


def execute_sql_preview(db_path: Path, sql: str, max_rows: int) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    ok, why = is_safe_readonly_sql(sql)
    if not ok:
        return [], [], f"UNSAFE_SQL:{why}"

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(sql.strip().rstrip(";"))
        cols = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in cur.fetchmany(max_rows)]
        return cols, rows, None
    except Exception as e:  # noqa: BLE001
        return [], [], f"SQL_EXEC_ERROR:{e}"
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Retrieve best matching seed SQL queries for a natural-language question.")

    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--seed-json", default="data/seed_questions.json", help="Seed JSON with decomposed_query blocks")
    source.add_argument("--candidate-json", help="Generic candidate JSON path")
    source.add_argument("--candidate-sqlite", help="SQLite DB path for candidate table")

    ap.add_argument("--candidate-table", default="query_library", help="SQLite table name when using --candidate-sqlite")
    ap.add_argument("--candidate-question-col", default="question", help="Question column in SQLite candidate table")
    ap.add_argument("--candidate-sql-col", default="sql", help="SQL column in SQLite candidate table")
    ap.add_argument("--candidate-id-col", default="id", help="ID column in SQLite candidate table")

    qsrc = ap.add_mutually_exclusive_group(required=True)
    qsrc.add_argument("--question", help="Input natural-language question")
    qsrc.add_argument("--question-json", help="JSON dataset containing questions")

    ap.add_argument("--question-index", type=int, default=0, help="Question index when using --question-json")
    ap.add_argument(
        "--question-key",
        default="natural_question",
        help="Question key in --question-json rows (fallbacks: natural_question/question/original_question)",
    )

    ap.add_argument("--top-k", type=int, default=5, help="Top-k candidates to return")
    ap.add_argument("--db-path", help="Optional SQLite DB path to execute top-ranked SQL for preview")
    ap.add_argument("--preview-rows", type=int, default=5, help="Max preview rows when --db-path is provided")
    ap.add_argument("--output-json", help="Optional output JSON path with ranked results")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.question:
        query_text = args.question.strip()
    else:
        query_text = load_question_from_dataset(Path(args.question_json), args.question_index, args.question_key)

    if args.candidate_sqlite:
        candidates = load_candidates_from_sqlite(
            db_path=Path(args.candidate_sqlite),
            table=args.candidate_table,
            question_col=args.candidate_question_col,
            sql_col=args.candidate_sql_col,
            id_col=args.candidate_id_col or None,
        )
    else:
        source_path = Path(args.candidate_json) if args.candidate_json else Path(args.seed_json)
        candidates = load_candidates_from_seed_json(source_path)

    if not candidates:
        raise SystemExit("No candidates loaded. Check input source format/path.")

    ranked = rank_candidates(query_text, candidates, top_k=args.top_k)

    print("=" * 90)
    print("INPUT QUESTION")
    print(query_text)
    print("=" * 90)
    print(f"Candidates loaded: {len(candidates)}")
    print(f"Top-k: {len(ranked)}")
    print("=" * 90)

    for m in ranked:
        print(f"Rank {m.rank} | total={m.total_score:.4f}")
        print(
            "  components: "
            f"lexical={m.lexical_score:.4f}, char={m.char_score:.4f}, "
            f"literal={m.literal_score:.4f}, operator={m.operator_score:.4f}, column={m.column_score:.4f}"
        )
        print(f"  candidate_id: {m.candidate.candidate_id}")
        print(f"  source: {m.candidate.source}")
        if m.candidate.parent_question:
            print(f"  parent_question: {m.candidate.parent_question}")
        print(f"  candidate_question: {m.candidate.question}")
        print(f"  candidate_sql: {m.candidate.sql}")
        print("-" * 90)

    preview = None
    if args.db_path and ranked:
        best = ranked[0]
        cols, rows, err = execute_sql_preview(Path(args.db_path), best.candidate.sql, max_rows=args.preview_rows)
        preview = {
            "candidate_id": best.candidate.candidate_id,
            "columns": cols,
            "rows": rows,
            "error": err,
        }
        print("TOP-1 SQL PREVIEW")
        print(f"candidate_id: {best.candidate.candidate_id}")
        if err:
            print(f"error: {err}")
        else:
            print(f"columns: {cols}")
            print(f"rows (max {args.preview_rows}):")
            for r in rows:
                print(f"  {r}")
        print("=" * 90)

    if args.output_json:
        payload = {
            "input_question": query_text,
            "ranked_results": [
                {
                    **asdict(m),
                    "candidate": asdict(m.candidate),
                }
                for m in ranked
            ],
            "top1_sql_preview": preview,
            "weights": {
                "lexical": W_LEXICAL,
                "char": W_CHAR,
                "literal": W_LITERAL,
                "operator": W_OPERATOR,
                "column": W_COLUMN,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved ranked results to: {out_path}")


if __name__ == "__main__":
    main()
