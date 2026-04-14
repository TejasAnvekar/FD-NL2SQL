#!/usr/bin/env python3
"""Format and evaluate one or more LLM run directories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FORMATTER = THIS_DIR / "batch_format_llm_predictions_like_ground_truth.py"
TABULAR_EVAL = PROJECT_ROOT / "eval_run_baselines_v3.py"
DERIVED_EVAL = PROJECT_ROOT / "eval_run_baselines_derived.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run the standard formatting + tabular eval + derived eval pipeline "
            "for one or more run directories."
        )
    )
    ap.add_argument("--run_dirs", nargs="+", required=True, help="Run directory path(s) to evaluate.")
    ap.add_argument("--skip_format", type=int, default=0, help="Skip the formatting step if already done.")
    ap.add_argument("--python_bin", default=sys.executable, help="Python executable to use.")
    return ap.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_dirs = [str(Path(p).expanduser().resolve()) for p in args.run_dirs]
    python_bin = args.python_bin

    if not args.skip_format:
        run_cmd([python_bin, str(FORMATTER), "--run_dirs", *run_dirs])

    run_cmd([python_bin, str(TABULAR_EVAL), "--run_dirs", *run_dirs])
    run_cmd([python_bin, str(DERIVED_EVAL), "--run_dirs", *run_dirs])


if __name__ == "__main__":
    main()
