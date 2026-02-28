#!/usr/bin/env python3
"""ChatGPT-friendly wrapper for full decompose->retrieve->synthesize pipeline.

Goals:
- Easy one-question testing
- GPT-ready batch processing
- Keep original orchestrator unchanged; this wrapper delegates to it
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _append(cmd: List[str], flag: str, value: Optional[str]) -> None:
    if value is None:
        return
    s = str(value).strip()
    if not s:
        return
    cmd.extend([flag, s])


def _resolve_api_key(args: argparse.Namespace) -> str:
    if str(args.api_key).strip():
        return str(args.api_key).strip()
    env_name = str(args.api_key_env).strip()
    if not env_name:
        raise SystemExit("Provide --api-key or --api-key-env for GPT/OpenAI usage.")
    key = os.getenv(env_name, "").strip()
    if not key:
        raise SystemExit(f"API key env var is empty or missing: {env_name}")
    return key


def _default_tag(mode: str, start: int, limit: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if mode == "single":
        return f"single_{ts}"
    if limit == -1:
        return f"batch_{start}_all_{ts}"
    end = start + max(0, limit) - 1
    return f"batch_{start}_{end}_{ts}"


def _run(cmd: List[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    root = _project_root()
    default_base_script = root / "method" / "orchestrate_decompose_retrieve_synthesize.py"

    ap = argparse.ArgumentParser(
        description="Refactored full pipeline launcher (single question + GPT batch ready)."
    )

    ap.add_argument("--mode", choices=["single", "batch"], default="single")

    # Question source
    ap.add_argument("--question", default="", help="Ad-hoc question text for --mode single")
    ap.add_argument("--question-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--question-index", type=int, default=0)
    ap.add_argument("--question-key", default="natural_question")
    ap.add_argument("--id-key", default="item_id")
    ap.add_argument("--start-index", type=int, default=0, help="Used in --mode batch")
    ap.add_argument("--limit", type=int, default=500, help="Used in --mode batch; -1 for all remaining")

    # Retrieval source
    ap.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    ap.add_argument("--candidate-json", default="")
    ap.add_argument("--candidate-sqlite", default="")

    # Prompt/schema
    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--decompose-prompt-file", default=str(root / "method" / "prompt" / "decompose_prompt.txt"))
    ap.add_argument(
        "--decompose-examples-file",
        default=str(root / "method" / "prompt" / "decompose_examples.txt"),
    )
    ap.add_argument(
        "--synthesis-prompt-file",
        default=str(root / "method" / "prompt" / "synthesis_prompt.txt"),
    )

    # Retrieval controls
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieval-per-decomp", type=int, default=5)
    ap.add_argument("--max-decomposed-queries", type=int, default=8)
    ap.add_argument(
        "--sbert-model",
        default=(
            "/mnt/shared/shared_hf_home/hub/models--google--embeddinggemma-300m/"
            "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
        ),
    )
    ap.add_argument("--sbert-device", default="")
    ap.add_argument("--sbert-batch-size", type=int, default=64)

    # GPT/OpenAI-compatible defaults
    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model-name", default="gpt-5-nano")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-retries", type=int, default=3)

    # Generation controls
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decompose-max-new-tokens", type=int, default=384)
    ap.add_argument("--synth-max-new-tokens", type=int, default=512)

    ap.add_argument("--decompose-gen-batch-size", type=int, default=32)
    ap.add_argument("--decompose-batch-concurrency", type=int, default=8)
    ap.add_argument("--synth-gen-batch-size", type=int, default=32)
    ap.add_argument("--synth-batch-concurrency", type=int, default=8)

    ap.add_argument("--synth-evidence-per-decomp", type=int, default=5)
    ap.add_argument("--synth-include-global-merged", type=int, default=0)
    ap.add_argument("--synth-where-only-evidence", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["none", "structured"], default="none")
    ap.add_argument("--logprobs", type=int, default=0)

    # Eval/acceptance
    ap.add_argument("--run-eval", type=int, default=1)
    ap.add_argument("--gt-json", default=str(root / "data" / "natural_question_1500.json"))
    ap.add_argument("--eval-script", default=str(root / "eval_run_baselines_v2.py"))
    ap.add_argument(
        "--acceptance-config",
        default=str(root / "method" / "decompose_method" / "seed_acceptance_rules.config"),
    )

    # Wrapper/runtime
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--base-script", default=str(default_base_script))
    ap.add_argument("--output-dir", default=str(root / "results" / "chatgpt_ready_pipeline_v2"))
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--skip-exec", type=int, default=1)
    ap.add_argument("--dry-run", type=int, default=0)

    # Staged seed-growth mode (forwarded to base orchestrator for parity)
    ap.add_argument("--staged-batch-mode", type=int, default=0)
    ap.add_argument("--stage-size", type=int, default=500)
    ap.add_argument("--stage-count", type=int, default=3)
    ap.add_argument("--batch-root", default=str(root / "method" / "batch"))
    ap.add_argument("--seed-copy-name", default="seed_working.json")
    ap.add_argument("--append-accepted-to-seed", type=int, default=1)
    ap.add_argument("--resume-staged", type=int, default=1)
    ap.add_argument("--checkpoint-path", default="")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.staged_batch_mode) == 1 and args.mode != "batch":
        raise SystemExit("--staged-batch-mode=1 requires --mode batch")

    output_dir = Path(args.output_dir)
    if not bool(int(args.dry_run)):
        output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = str(args.run_tag).strip() or _default_tag(args.mode, int(args.start_index), int(args.limit))
    api_key = _resolve_api_key(args) if (args.backend == "openai_compat" and not bool(int(args.dry_run))) else "dummy"

    effective_logprob_mode = str(args.logprob_mode)
    effective_logprobs = int(args.logprobs)
    if args.backend == "openai_compat":
        if effective_logprob_mode != "none" or effective_logprobs != 0:
            print("[warn] openai_compat run requested logprobs; forcing logprob-mode=none and logprobs=0 for GPT safety")
        effective_logprob_mode = "none"
        effective_logprobs = 0

    cmd: List[str] = [str(args.python_exe), str(args.base_script)]

    # Mode wiring
    if args.mode == "single":
        if str(args.question).strip():
            cmd.extend(["--question", str(args.question).strip()])
        else:
            cmd.extend([
                "--question-json",
                str(args.question_json),
                "--question-index",
                str(int(args.question_index)),
                "--question-key",
                str(args.question_key),
                "--id-key",
                str(args.id_key),
                "--batch-mode",
                "0",
            ])
    else:
        cmd.extend([
            "--question-json",
            str(args.question_json),
            "--question-key",
            str(args.question_key),
            "--id-key",
            str(args.id_key),
            "--batch-mode",
            "1",
            "--start-index",
            str(int(args.start_index)),
            "--limit",
            str(int(args.limit)),
        ])

    # Candidate source
    if str(args.candidate_sqlite).strip():
        cmd.extend(["--candidate-sqlite", str(args.candidate_sqlite)])
    elif str(args.candidate_json).strip():
        cmd.extend(["--candidate-json", str(args.candidate_json)])
    else:
        cmd.extend(["--seed-json", str(args.seed_json)])

    cmd.extend([
        "--schema-json",
        str(args.schema_json),
        "--decompose-prompt-file",
        str(args.decompose_prompt_file),
        "--decompose-examples-file",
        str(args.decompose_examples_file),
        "--synthesis-prompt-file",
        str(args.synthesis_prompt_file),
        "--top-k",
        str(int(args.top_k)),
        "--retrieval-per-decomp",
        str(int(args.retrieval_per_decomp)),
        "--max-decomposed-queries",
        str(int(args.max_decomposed_queries)),
        "--sbert-model",
        str(args.sbert_model),
        "--sbert-batch-size",
        str(int(args.sbert_batch_size)),
        "--backend",
        str(args.backend),
        "--api-base",
        str(args.api_base),
        "--api-key",
        str(api_key),
        "--model-name",
        str(args.model_name),
        "--timeout",
        str(float(args.timeout)),
        "--num-retries",
        str(int(args.num_retries)),
        "--decompose-max-new-tokens",
        str(int(args.decompose_max_new_tokens)),
        "--synth-max-new-tokens",
        str(int(args.synth_max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--seed",
        str(int(args.seed)),
        "--decompose-gen-batch-size",
        str(int(args.decompose_gen_batch_size)),
        "--decompose-batch-concurrency",
        str(int(args.decompose_batch_concurrency)),
        "--synth-gen-batch-size",
        str(int(args.synth_gen_batch_size)),
        "--synth-batch-concurrency",
        str(int(args.synth_batch_concurrency)),
        "--synth-evidence-per-decomp",
        str(int(args.synth_evidence_per_decomp)),
        "--synth-include-global-merged",
        str(int(args.synth_include_global_merged)),
        "--synth-where-only-evidence",
        str(int(args.synth_where_only_evidence)),
        "--logprob-mode",
        str(effective_logprob_mode),
        "--logprobs",
        str(int(effective_logprobs)),
        "--skip-exec",
        str(int(args.skip_exec)),
        "--run-eval",
        str(int(args.run_eval)),
        "--gt-json",
        str(args.gt_json),
        "--eval-script",
        str(args.eval_script),
        "--acceptance-config",
        str(args.acceptance_config),
        "--output-dir",
        str(output_dir),
        "--run-tag",
        str(run_tag),
        "--staged-batch-mode",
        str(int(args.staged_batch_mode)),
        "--stage-size",
        str(int(args.stage_size)),
        "--stage-count",
        str(int(args.stage_count)),
        "--batch-root",
        str(args.batch_root),
        "--seed-copy-name",
        str(args.seed_copy_name),
        "--append-accepted-to-seed",
        str(int(args.append_accepted_to_seed)),
        "--resume-staged",
        str(int(args.resume_staged)),
    ])
    _append(cmd, "--checkpoint-path", str(args.checkpoint_path))

    _append(cmd, "--sbert-device", args.sbert_device)

    print("Output dir:", output_dir)
    print("Run tag:", run_tag)
    if args.backend == "openai_compat":
        print(f"Logprobs: disabled (logprob-mode={effective_logprob_mode}, logprobs={effective_logprobs})")
    else:
        print(f"Logprobs: mode={effective_logprob_mode}, logprobs={effective_logprobs}")
    _run(cmd, dry_run=bool(int(args.dry_run)))

    print("\nDone")
    print("Run outputs root:", output_dir)
    print("Run tag:", run_tag)


if __name__ == "__main__":
    main()
