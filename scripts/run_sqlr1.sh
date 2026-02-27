#!/bin/bash

# =====================================================
# GPU SELECTION
# =====================================================

if [ ! -z "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
TP_SIZE=$NUM_GPUS

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME="MPX0222forHF/SQL-R1-14B"
GPU_MEM="${GPU_MEM:-0.85}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

PROJECT_DIR="/mnt/data1/tanvekar/MAYO_AIM2"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/sql_r1_vllm}"
mkdir -p "$LOG_ROOT"

RUN_ID=$(date +%Y%m%d_%H%M%S)

# Data + prompts
INPUT_JSON="${INPUT_JSON:-${PROJECT_DIR}/data/natural_question_1500.json}"
SCHEMA_JSON="${SCHEMA_JSON:-${PROJECT_DIR}/data/schema.json}"

PROMPT_STYLE="${PROMPT_STYLE:-sql_only}"   # use "cot" if you want chain-of-thought
USE_PYDANTIC="${USE_PYDANTIC:-1}"
LOGPROB_MODE="${LOGPROB_MODE:-structured}"

BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_CONCURRENCY="${BATCH_CONCURRENCY:-16}"

RUN_LOG_DIR="${LOG_ROOT}/${PROMPT_STYLE}_${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Starting vLLM API SQL-R1 Runner"
echo "Prompt style: $PROMPT_STYLE"
echo "GPU ids: $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Run ID: $RUN_ID"
echo "Pydantic schema: $USE_PYDANTIC"
echo "Logprob mode: $LOGPROB_MODE"
echo "=========================================="

# =====================================================
# SERVER HEALTH CHECK
# =====================================================

check_server_ready() {
    for i in {1..60}; do
        if curl -s -f "${VLLM_API_BASE%/v1}/health" > /dev/null 2>&1; then
            echo "✅ vLLM server ready"
            return 0
        fi
        echo "Waiting for server... ($i/60)"
        sleep 2
    done
    echo "❌ Server failed to start"
    return 1
}

cleanup() {
    echo ""
    echo "Stopping vLLM..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
    fi
}

trap cleanup EXIT INT TERM

# =====================================================
# START vLLM SERVER
# =====================================================

echo "Launching vLLM OpenAI-compatible server..."

vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TP_SIZE \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization $GPU_MEM \
    --max-logprobs 50 \
    --chat-template-content-format string \
    > "${RUN_LOG_DIR}/vllm_server.log" 2>&1 &

VLLM_PID=$!

if ! check_server_ready; then
    echo "Check log: ${RUN_LOG_DIR}/vllm_server.log"
    exit 1
fi

# =====================================================
# RUN SQL-R1 INFERENCE
# =====================================================

echo "Running SQL-R1 batch inference..."

python run_sqlr1.py \
    --api_base "$VLLM_API_BASE" \
    --api_key "dummy" \
    --model "$MODEL_NAME" \
    --input_json "$INPUT_JSON" \
    --schema_json "$SCHEMA_JSON" \
    --output_json "${RUN_LOG_DIR}/generated_sql.json" \
    --start "${START:-0}" \
    --limit "${LIMIT:-1500}" \
    --question_keys "${QUESTION_KEYS:-natural_question,question,original_question,new_question}" \
    --table_name "${TABLE_NAME:-clinical_trials}" \
    --max_tokens 512 \
    --temperature 0.0 \
    --top_p 1.0 \
    --timeout 120 \
    --num_retries 2 \
    --batch_size "$BATCH_SIZE" \
    --batch_concurrency "$BATCH_CONCURRENCY" \
    --prompt_style "$PROMPT_STYLE" \
    --use_pydantic_schema "$USE_PYDANTIC" \
    --logprob_mode "$LOGPROB_MODE" \
    --prompt_logprobs "${PROMPT_LOGPROBS:-0}" \
    --top_logprobs "${TOP_LOGPROBS:-0}"

echo "=========================================="
echo "Run complete."
echo "Logs:       $RUN_LOG_DIR"
echo "Outputs:    ${RUN_LOG_DIR}/generated_sql.json"
echo "=========================================="