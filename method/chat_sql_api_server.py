#!/usr/bin/env python3
"""HTTP API wrapper for one-question orchestrator.

Endpoint:
- POST /api/chat_sql with JSON body {"question": "...", "item_id": "...?"}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from provider_config import PROVIDER_CHOICES, resolve_openai_compat


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    default_api_base = "http://127.0.0.1:8000/v1"
    default_model_name = "qwen3-30b-local"

    ap = argparse.ArgumentParser(description="Serve /api/chat_sql by invoking the single-question orchestrator.")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--cors-origin", default="*")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--timeout-seconds", type=int, default=300)

    ap.add_argument(
        "--orchestrator-script",
        default=str(root / "method" / "orchestrate_single_question_chat.py"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(root / "results" / "chat_api_runs"),
    )

    # Forwarded orchestrator config
    ap.add_argument("--backend", choices=["openai_compat", "vllm_local"], default="openai_compat")
    ap.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default="local_vllm",
        help="Provider preset for backend=openai_compat (gemini/openai/openrouter/etc).",
    )
    ap.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable name to read API key from (overrides provider default env).",
    )
    ap.add_argument("--api-base", default=default_api_base)
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-name", default=default_model_name)
    ap.add_argument("--seed-json", default=str(root / "data" / "seed_questions.json"))
    ap.add_argument("--schema-json", default=str(root / "data" / "schema.json"))
    ap.add_argument("--decompose-prompt-file", default=str(root / "method" / "prompt" / "decompose_prompt.txt"))
    ap.add_argument("--decompose-examples-file", default=str(root / "method" / "prompt" / "decompose_examples.txt"))
    ap.add_argument("--synthesis-prompt-file", default=str(root / "method" / "prompt" / "synthesis_prompt.txt"))
    ap.add_argument("--db-path", default=str(root / "data" / "database.db"))
    ap.add_argument("--skip-exec", type=int, default=1)
    ap.add_argument("--run-eval", type=int, default=0)
    ap.add_argument("--use-pydantic-schema", type=int, default=1)
    ap.add_argument("--logprob-mode", choices=["structured", "none"], default="structured")

    return ap.parse_args()


class ChatSQLHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, cfg: argparse.Namespace):
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg


class Handler(BaseHTTPRequestHandler):
    server: ChatSQLHTTPServer  # type: ignore[assignment]

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.server.cfg.cors_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": f"Not found: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/chat_sql":
            self._send_json(404, {"error": f"Not found: {path}"})
            return

        try:
            cl = int(self.headers.get("Content-Length", "0"))
        except Exception:
            cl = 0

        raw = self.rfile.read(cl) if cl > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        question = str(body.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "Missing 'question' in request body"})
            return

        item_id = str(body.get("item_id") or "").strip()
        run_tag = str(body.get("run_tag") or "").strip()
        if not run_tag:
            run_tag = f"chat_{uuid.uuid4().hex[:12]}"

        cfg = self.server.cfg
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_json = out_dir / f"{run_tag}.chat_response.json"

        cmd: List[str] = [
            str(cfg.python_bin),
            str(cfg.orchestrator_script),
            "--question",
            question,
            "--output-dir",
            str(out_dir),
            "--run-tag",
            run_tag,
            "--output-json",
            str(output_json),
            "--backend",
            str(cfg.backend),
            "--provider",
            str(cfg.provider),
            "--api-base",
            str(cfg.api_base),
            "--api-key",
            str(cfg.api_key),
            "--api-key-env",
            str(cfg.api_key_env),
            "--model-name",
            str(cfg.model_name),
            "--seed-json",
            str(cfg.seed_json),
            "--schema-json",
            str(cfg.schema_json),
            "--decompose-prompt-file",
            str(cfg.decompose_prompt_file),
            "--decompose-examples-file",
            str(cfg.decompose_examples_file),
            "--synthesis-prompt-file",
            str(cfg.synthesis_prompt_file),
            "--db-path",
            str(cfg.db_path),
            "--skip-exec",
            str(int(cfg.skip_exec)),
            "--run-eval",
            str(int(cfg.run_eval)),
            "--use-pydantic-schema",
            str(int(cfg.use_pydantic_schema)),
            "--logprob-mode",
            str(cfg.logprob_mode),
        ]
        if item_id:
            cmd.extend(["--item-id", item_id])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(cfg.timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": "Orchestrator timed out"})
            return
        except Exception as e:  # noqa: BLE001
            self._send_json(500, {"error": f"Failed to invoke orchestrator: {e}"})
            return

        if proc.returncode != 0:
            self._send_json(
                500,
                {
                    "error": "Orchestrator failed",
                    "returncode": proc.returncode,
                    "stdout": (proc.stdout or "")[-4000:],
                    "stderr": (proc.stderr or "")[-4000:],
                },
            )
            return

        if not output_json.exists():
            self._send_json(
                500,
                {
                    "error": "Orchestrator completed but output JSON not found",
                    "output_json": str(output_json),
                },
            )
            return

        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            self._send_json(500, {"error": f"Failed reading orchestrator output: {e}"})
            return

        self._send_json(200, payload)


def main() -> None:
    args = parse_args()

    if args.backend == "openai_compat":
        try:
            args.api_base, args.api_key, args.model_name, provider_meta = resolve_openai_compat(
                provider=str(args.provider),
                api_base=str(args.api_base),
                api_key=str(args.api_key),
                model_name=str(args.model_name),
                api_key_env=str(args.api_key_env),
                local_default_api_base="http://127.0.0.1:8000/v1",
                local_default_model_name="qwen3-30b-local",
            )
            args.provider = str(provider_meta.get("provider") or args.provider)
        except ValueError as e:  # noqa: BLE001
            raise SystemExit(str(e))

    server = ChatSQLHTTPServer((args.host, int(args.port)), Handler, args)
    print(f"Chat SQL API listening on http://{args.host}:{args.port}")
    print(f"Orchestrator script: {args.orchestrator_script}")
    if args.backend == "openai_compat":
        print(f"Provider: {args.provider} | api_base={args.api_base} | model={args.model_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
