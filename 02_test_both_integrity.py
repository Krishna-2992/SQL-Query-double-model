import torch
import json
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model.eval()

SQL_MODEL_NAME = "prem-research/prem-1B-SQL"

sql_tokenizer = AutoTokenizer.from_pretrained(SQL_MODEL_NAME)

sql_model = AutoModelForCausalLM.from_pretrained(
    SQL_MODEL_NAME,
    torch_dtype=torch.float16
).to("cuda")

sql_model.eval()

print("Reasoning model device:", next(model.parameters()).device)
print("SQL model device:", next(sql_model.parameters()).device)

ALLOWED_INTENTS = [
    "CREATE_TABLE",
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "ALTER",
    "DROP"
]

REASONING_SCHEMA = """
Fields:
- intent: one of CREATE_TABLE, SELECT, INSERT, UPDATE, DELETE, ALTER, DROP
- table: lowercase table name
- set: key-value pairs to modify or insert
- where: key-value filter conditions
- followup: boolean

Output format:
A single JSON object with keys:
intent, table, set, where, followup
"""

def build_prompt(user_request: str) -> str:
    return f"""
You are a STRICT SQL intent classifier and extractor.

Rules:
- Output EXACTLY ONE JSON object
- Do NOT include schema or examples
- Do NOT repeat instructions
- Do NOT explain anything
- Do NOT use markdown
- Do NOT invent columns unless user mentions them
- intent MUST be one of: {", ".join(ALLOWED_INTENTS)}
- table MUST be lowercase
- set = values to modify or insert
- where = filter conditions
- followup = false unless user clearly refers to previous query

JSON format:
{REASONING_SCHEMA}

User request:
{user_request}

JSON:
"""

def extract_json(text: str):
    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None

def normalize_reasoning(result: dict) -> dict:
    if not result:
        return None

    # ---- intent normalization ----
    intent_map = {
        "update": "UPDATE",
        "create": "INSERT",
        "create_table": "CREATE_TABLE",
        "delete": "DELETE",
        "remove": "DELETE",
        "retrieve": "SELECT",
        "select": "SELECT",
        "insert": "INSERT"
    }

    raw_intent = result.get("intent", "").lower().strip()
    result["intent"] = intent_map.get(raw_intent, raw_intent.upper())

    # ---- table normalization ----
    if "table" in result and isinstance(result["table"], str):
        result["table"] = result["table"].lower()

    # ---- ensure required fields ----
    result.setdefault("set", {})
    result.setdefault("where", {})
    result.setdefault("followup", False)

    # ---- repair where clause ----
    if not result["where"] and isinstance(result.get("set"), dict):
        # move identifier-like fields to WHERE
        for k in list(result["set"].keys()):
            if k.endswith("_id") or k == "id":
                result["where"][k] = result["set"].pop(k)

    return result

def run_reasoning(user_request: str):
    prompt = build_prompt(user_request)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- RAW OUTPUT ---")
    print(decoded)

    parsed = extract_json(decoded)
    parsed = normalize_reasoning(parsed)

    print("\n--- PARSED JSON ---")
    if parsed:
        print(json.dumps(parsed, indent=2))
    else:
        print("❌ Invalid JSON extracted")

    print(f"\n⏱️ Inference time: {elapsed:.3f} seconds")

    return parsed

def run_sql_generation(sql_prompt: str) -> str:
    inputs = sql_tokenizer(
        sql_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")

    with torch.no_grad():
        output = sql_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    return sql_tokenizer.decode(output[0], skip_special_tokens=True)

def build_sql_prompt_from_reasoning(reasoning: dict) -> str:
    intent = reasoning["intent"]
    table = reasoning["table"]
    where = reasoning.get("where", {})

    if intent == "SELECT":
        if where:
            conditions = " AND ".join(
                f"{k} = '{v}'" for k, v in where.items()
            )
            return f"SELECT * FROM {table} WHERE {conditions};"
        else:
            return f"SELECT * FROM {table};"

    if intent == "CREATE_TABLE":
        return f"Create a SQL table for {table} using appropriate columns."

    # For non‑SELECT, we should not call Prem
    return None

if __name__ == "__main__":
    tests = [
        # "Increase salary to 70000 for employee with id 5",
        "Show all customers from India",
        # "Create a table to store orders with date and amount",
        # "Delete user where email = test@gmail.com",
        # "Add a new product with name phone and price 500",
        # "Update order status to shipped where order id is 10"
    ]

    for t in tests:
        print("\n" + "=" * 80)
        print("USER:", t)

        reasoning = run_reasoning(t)

        if not reasoning or reasoning["intent"] == "UNKNOWN":
            print("❌ Could not reason about query")
            continue

        print("🚀reasining🚀")
        print(reasoning)

        sql_prompt = build_sql_prompt_from_reasoning(reasoning)

        print("🚀sql_prompt🚀")
        print(sql_prompt)

        if sql_prompt:
            print("\n--- SQL PROMPT (to Prem) ---")
            print(sql_prompt)

            sql = run_sql_generation(sql_prompt)

            print("\n--- FINAL SQL (from Prem) ---")
            print(sql)
        else:
            print("\n--- DETERMINISTIC SQL (no Prem) ---")
            print(reasoning)

