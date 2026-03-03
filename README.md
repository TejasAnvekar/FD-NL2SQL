# FD-NL2SQL Local Setup Guide

This guide explains how to run FD-NL2SQL locally end-to-end:
- method scripts (single question / batch)
- chat backend API
- frontend UI

## 1) Prerequisites
- Python: 3.10+ (3.11 recommended)
- Node.js: 18+ and npm
- Git
- OpenAI API key

## 2) Clone and Enter Project

### Linux/macOS (bash)
```bash
git clone https://github.com/TejasAnvekar/FD-NL2SQL.git FD-NL2SQL
cd FD-NL2SQL
```

### Windows (PowerShell)
```powershell
git clone https://github.com/TejasAnvekar/FD-NL2SQL.git FD-NL2SQL
Set-Location FD-NL2SQL
```

## 3) Python Environment (Conda)

### Linux/macOS (bash)
```bash
conda create -n fdnl2sql python=3.11 -y
conda activate fdnl2sql
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" openai pydantic numpy
```

### Windows (PowerShell)
```powershell
conda create -n fdnl2sql python=3.11 -y
conda activate fdnl2sql
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" openai pydantic numpy
```

OpenAI embeddings are the default retrieval backend in this repo.

Optional SBERT fallback dependency:
```bash
pip install sentence-transformers
```

If your local pipeline path needs extra packages (for eval/model paths), install them in the same Conda env.

## 4) Configure Environment Variables

Create a `.env` file at project root with this content:

```env
OPENAI_API_KEY=your_openai_api_key_here
CHAT_API_KEY=your_openai_api_key_here
CHAT_API_BASE=https://api.openai.com/v1
CHAT_BACKEND=openai_compat
CHAT_MODEL_NAME=gpt-5-nano
CHAT_LOGPROB_MODE=none
CHAT_LOGPROBS=0
CHAT_EMBED_BACKEND=openai
CHAT_EMBED_MODEL=text-embedding-3-small
CHAT_EMBED_API_BASE=https://api.openai.com/v1
CHAT_EMBED_API_KEY=your_openai_api_key_here
CHAT_EMBED_BATCH_SIZE=128
CHAT_API_HOST=127.0.0.1
CHAT_API_PORT=8181
CHAT_CORS_ALLOW_ORIGINS=http://localhost:5173
```

Load `.env` into the current shell:

### Linux/macOS (bash)
```bash
set -a
source .env
set +a
```

### Windows (PowerShell)
```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $name, $value = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim().Trim('"'), 'Process')
}
```

## 5) Verify Required Data Files

Ensure these files exist:
- `data/schema.json`
- `data/database.db`
- `data/seed_questions.json`
- `data/natural_question_1500.json` (for batch/eval runs)

## 6) Run Chat Backend API

From project root:

### Linux/macOS
```bash
python chat_pipeline_api.py
```

### Windows (PowerShell)
```powershell
python .\chat_pipeline_api.py
```

If port `8181` is occupied, choose another port first:

### Linux/macOS
```bash
export CHAT_API_PORT=8181
python chat_pipeline_api.py
```

### Windows (PowerShell)
```powershell
$env:CHAT_API_PORT="8182"
python .\chat_pipeline_api.py
```

Health check:

### Linux/macOS
```bash
curl http://127.0.0.1:8181/health
```

### Windows (PowerShell)
```powershell
Invoke-RestMethod http://127.0.0.1:8181/health
```

Expected: JSON with `"ok": true`.

## 7) Run Frontend

In a second terminal:

### Linux/macOS
```bash
cd frontend
npm install
```

### Windows (PowerShell)
```powershell
Set-Location .\frontend
npm install
```

Create `frontend/.env.local` with:

```env
VITE_ORCHESTRATOR_PROXY_TARGET=http://127.0.0.1:8181
```

Start frontend:

```bash
npm run dev -- --host 0.0.0.0 --port 5173
```

If you are on a remote machine (SSH/VM/WSL), open the **Network** URL shown by Vite (for example `http://10.x.x.x:5173`).

If you are running directly on your local machine, open `http://localhost:5173`.

## 8) Run Method Without Frontend (Optional)

### Single question (chat orchestrator)

#### Linux/macOS
```bash
python method/orchestrate_single_question_chat.py \
  --question "Which colorectal trials had 3 or more arms and used a multikinase inhibitor as control?" \
  --backend openai_compat \
  --api-base https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model-name gpt-5-nano \
  --logprob-mode none \
  --logprobs 0 \
  --skip-exec 0
```

#### Windows (PowerShell)
```powershell
python .\method\orchestrate_single_question_chat.py `
  --question "Which colorectal trials had 3 or more arms and used a multikinase inhibitor as control?" `
  --backend openai_compat `
  --api-base https://api.openai.com/v1 `
  --api-key $env:OPENAI_API_KEY `
  --model-name gpt-5-nano `
  --logprob-mode none `
  --logprobs 0 `
  --skip-exec 0
```

### Batch pipeline

#### Linux/macOS
```bash
python method/orchestrate_decompose_retrieve_synthesize.py \
  --mode batch \
  --start-index 0 \
  --limit 10 \
  --backend openai_compat \
  --api-base https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model-name gpt-5-nano
```

#### Windows (PowerShell)
```powershell
python .\method\orchestrate_decompose_retrieve_synthesize.py `
  --mode batch `
  --start-index 0 `
  --limit 10 `
  --backend openai_compat `
  --api-base https://api.openai.com/v1 `
  --api-key $env:OPENAI_API_KEY `
  --model-name gpt-5-nano
```

## 9) Common Issues

### `Pipeline API error 405`
- Usually wrong endpoint path or stale frontend build.
- Confirm frontend calls `/api/chat-query` on the configured backend target.
- Confirm backend is running and CORS includes frontend origin.

### CORS blocked (`Access-Control-Allow-Origin` missing)
- Set `CHAT_CORS_ALLOW_ORIGINS=http://localhost:5173` in `.env`.
- Restart backend after changing env vars.

### `401 invalid_api_key`
- Check `OPENAI_API_KEY` / `CHAT_API_KEY` values.
- Avoid quoting mistakes or truncated keys.

### `Unsupported parameter: max_tokens`
- GPT-5 models may require `max_completion_tokens` in some callers.
- Use the current repo scripts (they already handle this where implemented).

## 10) Suggested Local Workflow

1. Start backend (`python chat_pipeline_api.py`).
2. Start frontend (`npm run dev -- --host 0.0.0.0 --port 5173`).
3. Validate `/health` in browser or curl.
4. Ask a test question in chat.
5. If errors occur, inspect browser Network tab + backend terminal logs.
