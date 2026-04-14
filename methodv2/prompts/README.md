# Prompts

This folder stores reusable prompt templates for the `methodv2` baselines.

Current files:

- `naive_llm_zero_shot_csv.txt`
  A reusable zero-shot prompt for `run_question_table_copy_llm_naive.py` that
  frames the task as row-wise table reasoning over a visible CSV table.

- `naive_llm_fewshot_csv.txt`
  A few-shot prompt for `run_question_table_copy_llm_naive.py` when the visible
  table is rendered as CSV.

- `naive_llm_cot_csv.txt`
  A zero-shot chain-of-thought-style prompt template for `run_question_table_copy_llm_naive.py` that asks the model to reason internally and return only final JSON.

Supported placeholders in prompt templates:

- `{{question}}`
- `{{table_csv_text}}`
- `{{table_text}}`

Example:

```bash
python3 /mnt/data1/srchowd3/FD-NL2SQL/methodv2/run_question_table_copy_llm_naive.py \
  --manifest_csv /mnt/data1/srchowd3/FD-NL2SQL/data/table_question_ground_truths_full/manifest.csv \
  --annotated_csv '/mnt/data1/srchowd3/FD-NL2SQL/data/cat3_query_sql_llm(2)_with_key_matches.csv' \
  --db_path /mnt/data1/srchowd3/FD-NL2SQL/data/database.db \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model_name llama-3.3-70b-instruct \
  --run_name question_table_copy_llm_naive_full_llama33_zeroshot \
  --limit 0 \
  --prompt_template_file /mnt/data1/srchowd3/FD-NL2SQL/methodv2/prompts/naive_llm_zero_shot_csv.txt \
  --max_in_flight 20
```
