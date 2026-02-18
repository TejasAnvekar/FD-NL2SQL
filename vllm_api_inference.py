import json
import os
import requests
import traceback
from typing import List, Dict
from pydantic import BaseModel

# ===============================
# CONFIG
# ===============================

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "google/gemma-3-27b-it"

INPUT_FILE = "data/dataset.jsonl"
OUTPUT_FILE = "results/generated_sql.jsonl"
SCHEMA_FILE = "data/schema.json"
TABLE_NAME = "clinical_trials"

MAX_TOKENS = 512
TEMPERATURE = 0.0

os.makedirs("results", exist_ok=True)

# ===============================
# LOAD SCHEMA
# ===============================

with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    schema_columns = json.load(f)

schema_string = f"Table: {TABLE_NAME}\nColumns:\n"
for col in schema_columns:
    schema_string += f'- "{col}"\n'

# ===============================
# OUTPUT JSON SCHEMA (Pydantic)
# ===============================

class SQLResponse(BaseModel):
    sql: str
    filters: Dict
    columns: List[str]

json_schema = SQLResponse.model_json_schema()

# ===============================
# PROMPT BUILDER
# ===============================

def build_messages(question: str):
    return [
        {
            "role": "system",
            "content": "You are an expert SQLite query generator."
        },
        {
            "role": "user",
            "content": f"""
Generate a valid SQLite query using only the schema below.

Schema:
{schema_string}

Question:
{question}

Return ONLY valid JSON.
"""
        }
    ]

# ===============================
# RESUME SUPPORT
# ===============================

processed = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        processed = sum(1 for _ in f)

print(f"Resuming from line {processed + 1}")

# ===============================
# GENERATION LOOP
# ===============================

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

    for line_number, line in enumerate(infile, start=1):

        if line_number <= processed:
            continue

        try:
            record = json.loads(line.strip())
            question = record.get("question", "").strip()

            if not question:
                record["error"] = "Empty question"
                outfile.write(json.dumps(record) + "\n")
                outfile.flush()
                os.fsync(outfile.fileno())
                continue

            payload = {
                "model": MODEL_NAME,
                "messages": build_messages(question),
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SQLResponse",
                        "schema": json_schema
                    }
                }
            }

            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = response.json()

            raw_text = result["choices"][0]["message"]["content"]
            record["raw_model_output"] = raw_text

            try:
                parsed_json = json.loads(raw_text)
                record["pred_sql"] = parsed_json.get("sql", "")
                record["pred_filters"] = parsed_json.get("filters", {})
                record["pred_columns"] = parsed_json.get("columns", [])
            except Exception as parse_error:
                record["pred_sql"] = ""
                record["pred_filters"] = {}
                record["pred_columns"] = []
                record["error"] = f"JSON parse error: {str(parse_error)}"

        except Exception as e:
            record = {
                "question": question if "question" in locals() else "",
                "pred_sql": "",
                "pred_filters": {},
                "pred_columns": [],
                "error": f"API error: {str(e)}",
                "traceback": traceback.format_exc()
            }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()
        os.fsync(outfile.fileno())

        print(f"[{line_number}] Saved.")

print("Done.")
