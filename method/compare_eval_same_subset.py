#!/usr/bin/env python3
"""Compare two eval JSON files on the same item-id subset.

Default subset behavior:
- Use item_ids where file A has non-empty pred_sql.
This matches staged runs where only a portion (e.g., 500) is predicted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two eval JSONs on the same subset.")
    ap.add_argument("--eval-a", required=True, help="Path to eval JSON A (subset source by default).")
    ap.add_argument("--eval-b", required=True, help="Path to eval JSON B.")
    ap.add_argument(
        "--subset",
        choices=["a_pred_nonempty", "b_pred_nonempty", "intersection_pred_nonempty"],
        default="a_pred_nonempty",
        help="How to pick item_id subset for comparison.",
    )
    ap.add_argument(
        "--show-disagreements",
        type=int,
        default=0,
        help="Show first N item_ids where exact_exec_match differs.",
    )
    return ap.parse_args()


def load_eval(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("per_item") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Expected dict with per_item list: {path}")
    return [r for r in rows if isinstance(r, dict)]


def has_nonempty_pred_sql(row: Dict[str, Any]) -> bool:
    v = row.get("pred_sql")
    return isinstance(v, str) and bool(v.strip())


def to_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = r.get("item_id")
        if k is None:
            continue
        out[str(k)] = r
    return out


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if not math.isnan(fv):
                xs.append(fv)
    if not xs:
        return None
    return sum(xs) / len(xs)


def metric_block(by_id: Dict[str, Dict[str, Any]], ids: Sequence[str]) -> Dict[str, Any]:
    n = len(ids)
    exact = [1.0 if bool(by_id.get(i, {}).get("exact_exec_match")) else 0.0 for i in ids]
    exec_ok = [1.0 if bool(by_id.get(i, {}).get("pred_exec_ok")) else 0.0 for i in ids]
    return {
        "n": n,
        "exact_exec_match_count": int(sum(exact)),
        "exact_exec_match_rate": (sum(exact) / n) if n else None,
        "pred_exec_ok_count": int(sum(exec_ok)),
        "pred_exec_ok_rate": (sum(exec_ok) / n) if n else None,
        "avg_f1": mean(by_id.get(i, {}).get("f1") for i in ids),
        "avg_sql_ast_similarity": mean(by_id.get(i, {}).get("sql_ast_similarity") for i in ids),
        "avg_chrf": mean(by_id.get(i, {}).get("chrf") for i in ids),
        "avg_rouge_l_f1": mean(by_id.get(i, {}).get("rouge_l_f1") for i in ids),
    }


def pick_subset_ids(
    a_by_id: Dict[str, Dict[str, Any]],
    b_by_id: Dict[str, Dict[str, Any]],
    mode: str,
) -> List[str]:
    if mode == "a_pred_nonempty":
        ids = [i for i, r in a_by_id.items() if has_nonempty_pred_sql(r)]
        return sorted(ids)
    if mode == "b_pred_nonempty":
        ids = [i for i, r in b_by_id.items() if has_nonempty_pred_sql(r)]
        return sorted(ids)

    # intersection_pred_nonempty
    a_ids = {i for i, r in a_by_id.items() if has_nonempty_pred_sql(r)}
    b_ids = {i for i, r in b_by_id.items() if has_nonempty_pred_sql(r)}
    return sorted(a_ids & b_ids)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main() -> None:
    args = parse_args()
    a_path = Path(args.eval_a)
    b_path = Path(args.eval_b)

    a_rows = load_eval(a_path)
    b_rows = load_eval(b_path)
    a_by = to_by_id(a_rows)
    b_by = to_by_id(b_rows)

    ids = pick_subset_ids(a_by, b_by, mode=str(args.subset))
    if not ids:
        raise SystemExit("No item_ids selected by subset rule.")

    # Keep only ids that exist in both files.
    common = [i for i in ids if i in a_by and i in b_by]
    missing_in_b = len(ids) - len(common)
    if not common:
        raise SystemExit("No overlapping item_ids between A and B for selected subset.")

    a_m = metric_block(a_by, common)
    b_m = metric_block(b_by, common)

    a_win = 0
    b_win = 0
    tie = 0
    disagreements: List[str] = []
    for i in common:
        ae = bool(a_by[i].get("exact_exec_match"))
        be = bool(b_by[i].get("exact_exec_match"))
        if ae and not be:
            a_win += 1
        elif be and not ae:
            b_win += 1
            disagreements.append(i)
        elif ae != be:
            disagreements.append(i)
        else:
            tie += 1

    print("Subset mode:", args.subset)
    print("Eval A:", a_path)
    print("Eval B:", b_path)
    print("Selected ids:", len(ids))
    print("Common ids used:", len(common))
    print("Missing in B from selected ids:", missing_in_b)
    print()

    print("[A metrics]")
    for k, v in a_m.items():
        print(f"{k}: {fmt(v)}")
    print()

    print("[B metrics]")
    for k, v in b_m.items():
        print(f"{k}: {fmt(v)}")
    print()

    print("[Head-to-head exact_exec_match]")
    print("A wins:", a_win)
    print("B wins:", b_win)
    print("Tie:", tie)

    n_show = max(0, int(args.show_disagreements))
    if n_show > 0:
        print()
        print(f"[First {n_show} disagreement item_ids]")
        for item_id in disagreements[:n_show]:
            ae = bool(a_by[item_id].get("exact_exec_match"))
            be = bool(b_by[item_id].get("exact_exec_match"))
            print(f"{item_id} | A={ae} B={be}")


if __name__ == "__main__":
    main()

