# methodv2

This folder contains the hidden-column experiments for `FD-NL2SQL`.

The main pipeline is:

1. Generate evidence SQL on a reduced schema with the hidden target column removed.
2. Execute that SQL.
3. Optionally enrich the returned rows with `NCT` and `Trial name`.
4. Optionally fetch ClinicalTrials.gov metadata by `NCT`.
5. Optionally run an LLM planner that decides which ClinicalTrials.gov fields are relevant for this specific question.
6. Ask the model to infer the hidden target value from the executed rows plus the planner-selected study evidence.
7. Save per-question tables, predictions, and checkpoints.

## Main Scripts

- `run_hidden_sql_then_infer_server.py`
  Main end-to-end runner using a local OpenAI-compatible vLLM server.

- `clinicaltrials_api.py`
  Small helper for fetching official ClinicalTrials.gov study records.

- `fetch_ctgov_metadata_for_question.py`
  Standalone row-level tester for ClinicalTrials.gov metadata retrieval.

- `question_table_exports.py`
  Writes per-question `final_table.csv` and `ground_truth_table.csv`.

## Start A Server

### Gemma 3 27B on one GPU

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared/shared_hf_home/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a \
  --served-model-name gemma-3-27b-it \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --dtype auto \
  --api-key EMPTY
```

### Llama 3.3 70B on two GPUs

This model usually needs at least 2 GPUs.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --served-model-name llama-3.3-70b-instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --dtype auto \
  --api-key EMPTY
```

### Check The Server

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

## Run The Main Pipeline

This runs on all usable questions, uses ClinicalTrials.gov metadata, and shows the source column in the prompt.

```bash
python3 /mnt/data1/srchowd3/FD-NL2SQL/methodv2/run_hidden_sql_then_infer_server.py \
  --csv_path '/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv' \
  --db_path /mnt/data1/srchowd3/FD-NL2SQL/data/database.db \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model_name llama-3.3-70b-instruct \
  --run_name llama33_70b_sql_then_infer_full_ctgov \
  --limit 0 \
  --prompt_source_column_mode shown \
  --use_ctgov_metadata 1 \
  --use_ctgov_hybrid_planner 1
```

If you want the prompt to hide the source column name:

```bash
--prompt_source_column_mode hidden
```

If you want to disable ClinicalTrials.gov metadata:

```bash
--use_ctgov_metadata 0
```

If you want to keep ClinicalTrials.gov enabled but skip the question-specific planner:

```bash
--use_ctgov_hybrid_planner 0
```

## Test A Single CSV Row

Example for row 153:

```bash
python3 /mnt/data1/srchowd3/FD-NL2SQL/methodv2/run_hidden_sql_then_infer_server.py \
  --csv_path '/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv' \
  --db_path /mnt/data1/srchowd3/FD-NL2SQL/data/database.db \
  --csv_row_number 153 \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model_name llama-3.3-70b-instruct \
  --run_name row153_ctgov_trial_guided \
  --limit 0 \
  --prompt_source_column_mode shown \
  --use_ctgov_metadata 1 \
  --use_ctgov_hybrid_planner 1
```

Dry run for setup/debugging:

```bash
python3 /mnt/data1/srchowd3/FD-NL2SQL/methodv2/run_hidden_sql_then_infer_server.py \
  --csv_path '/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv' \
  --db_path /mnt/data1/srchowd3/FD-NL2SQL/data/database.db \
  --csv_row_number 153 \
  --run_name row153_dry_run \
  --limit 0 \
  --prompt_source_column_mode shown \
  --dry_run
```

## Test ClinicalTrials.gov Retrieval Only

This does not run the LLM pipeline. It only executes the row SQL with `NCT` and `Trial name` added, then fetches study metadata from the official API.

```bash
python3 /mnt/data1/srchowd3/FD-NL2SQL/methodv2/fetch_ctgov_metadata_for_question.py \
  --csv_path '/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv' \
  --db_path /mnt/data1/srchowd3/FD-NL2SQL/data/database.db \
  --csv_row_number 153 \
  --run_name row153_ctgov
```

## Outputs

For a run named `my_run`, outputs go to:

```bash
/mnt/data1/srchowd3/FD-NL2SQL/methodv2/runs/my_run
```

Important files:

- `all_question_results.csv`
  One row per question with SQL, planner output, CTGov metadata, inference output, and file paths.

- `row_level_predictions.csv`
  One row per returned evidence row with predicted vs actual hidden value.

- `questions/`
  One folder per question.

- `questions/<question-folder>/final_table.csv`
  The final predicted table for that question.

- `questions/<question-folder>/ground_truth_table.csv`
  The ground-truth table for that question.

- `questions/<question-folder>/metadata.json`
  Metadata for that question.

- `questions/<question-folder>/ctgov_metadata.json`
  ClinicalTrials.gov metadata used for that question, when available.

## Resume / Checkpointing

The main runner supports resume if you reuse the same `--run_name`.

It writes checkpoint files as it goes:

- `question_results_checkpoint.jsonl`
- `row_level_checkpoint.jsonl`
- `sql_requests.jsonl`
- `sql_responses.jsonl`
- `planner_requests.jsonl`
- `planner_responses.jsonl`
- `inference_requests.jsonl`
- `inference_responses.jsonl`

If a run stops, rerun the exact same command with the same `--run_name` and it will skip completed `item_id`s.

`--dry_run` does not create resume checkpoints.
