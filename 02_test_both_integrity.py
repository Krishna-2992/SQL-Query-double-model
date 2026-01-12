import torch
import json
import re
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model.eval()

# ------ FOR WINDOWS ------
# SQL_MODEL_NAME = "prem-research/prem-1B-SQL"

# sql_tokenizer = AutoTokenizer.from_pretrained(SQL_MODEL_NAME)

# sql_model = AutoModelForCausalLM.from_pretrained(
#     SQL_MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="cpu"
# )

# sql_model.eval()

# ------ FOR MAC ------
OLLAMA_URL = "http://localhost:11434/api/generate"
PREM_MODEL = "anindya/prem1b-sql-ollama-fp116:latest"

def run_prem_sql(prompt: str) -> str:
    payload = {
        "model": PREM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()

    return resp.json()["response"].strip()




print("Reasoning model device:", next(model.parameters()).device)
# ------ FOR WINDOWS ------
# print("SQL model device:", next(sql_model.parameters()).device)

REASONING_SCHEMA = """
Fields:
- intent: SELECT | INSERT | UPDATE | DELETE | CREATE_TABLE | ALTER | DROP
- tables: list of tables involved
- columns: list of selected columns or expressions
- joins: list of join objects
    - type: INNER | LEFT | RIGHT
    - table: table name
    - on: join condition
- filters: list of conditions
    - column
    - operator (=, >, <, >=, <=, LIKE, IN, BETWEEN)
    - value
    - logical: AND | OR
- group_by: list of columns
- having: list of conditions
- order_by: list of columns with direction
    - column
    - direction: ASC | DESC
- limit: integer or null
- offset: integer or null
- set: key-value pairs (for INSERT / UPDATE)
- followup: boolean
"""

def build_prompt(user_request: str) -> str:
    return f"""
You are a SQL semantic parser.

Your task:
- Understand the user request
- Extract ALL relevant SQL components
- Preserve comparisons, ordering, grouping, limits, joins, and conditions

Rules:
- Output EXACTLY one JSON object
- Use only fields from the schema
- If a field is not mentioned, use null or empty list
- Do NOT generate SQL
- Do NOT explain anything
- Do NOT repeat this prompt in the response

Schema:
{REASONING_SCHEMA}

User request:
{user_request}

JSON:
"""

def extract_json(text: str):
    json_start = text.find("JSON:")
    if json_start == -1:
        return None

    start = text.find("{", json_start)
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
                except json.JSONDecodeError as e:
                    print("JSON decode error:", e)
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
            max_new_tokens=300,
            do_sample=False
        )

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- RAW OUTPUT ---")
    print(decoded)

    parsed = extract_json(decoded)

    print("🎲🎲🎲🎲🎲🎲🎲🎲🎲🎲")
    print(parsed)
    print("🎲🎲🎲🎲🎲🎲🎲🎲🎲🎲")
    parsed = normalize_reasoning(parsed)
    print("🛟🛟🛟🛟🛟🛟🛟🛟")
    print(parsed)
    print("🛟🛟🛟🛟🛟🛟🛟🛟")

    print("\n--- PARSED JSON ---")
    if parsed:
        print(json.dumps(parsed, indent=2))
    else:
        print("❌ Invalid JSON extracted")

    print(f"\n⏱️ Inference time: {elapsed:.3f} seconds")

    return parsed

# def run_sql_generation(sql_prompt: str) -> str:
#     inputs = sql_tokenizer(
#         sql_prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512
#     ).to("cuda")

#     with torch.no_grad():
#         output = sql_model.generate(
#             **inputs,
#             max_new_tokens=200,
#             do_sample=False
#         )

#     return sql_tokenizer.decode(output[0], skip_special_tokens=True)

# ------- FOR MAC -------
def run_sql_generation(sql_prompt: str) -> str:
    return run_prem_sql(sql_prompt)

def build_sql_prompt_from_reasoning(reasoning: dict) -> str:
    return f"""
You are an expert SQL generator.

Input is a JSON specification of a SQL query.
Generate the most accurate SQL query possible.

Rules:
- Use standard ANSI SQL
- Do not hallucinate tables or columns
- Do not explain anything
- Output only SQL

JSON:
{json.dumps(reasoning, indent=2)}

SQL:
"""

if __name__ == "__main__":
    tests = [
        # "Increase salary to 70000 for employee with id 5",
        # "Show all customers from India",
        # "Create a table to store orders with date and amount",
        # "Delete user where email = test@gmail.com",
        # "Add a new product with name phone and price 500",
        # "Update order status to shipped where order id is 10", 
        # "Show top 5 customers by total order value in 2023"
        "schema: User {id, name}, Order {order_id, price, user_id}, question: which user purchased things of more than 1000 rs."
    ]

    for t in tests:
        print("\n" + "=" * 80)
        print("USER:", t)

        reasoning = run_reasoning(t)

        if not reasoning or reasoning["intent"] == "UNKNOWN":
            print("❌ Could not reason about query")
            continue

        print("🚀reasoning🚀🚀🚀🚀🚀")
        print(reasoning)
        print("🚀🚀🚀🚀🚀🚀")

        sql_prompt = build_sql_prompt_from_reasoning(reasoning)

        print("🏀🏀🏀🏀🏀🏀🏀🏀🏀SQL prompt")
        print(sql_prompt)
        print("🏀🏀🏀🏀🏀🏀🏀🏀")

        if sql_prompt:
            # print("\n--- SQL PROMPT (to Prem) ---")
            # print(sql_prompt)

            sql = run_sql_generation(sql_prompt)

            print("\n--- FINAL SQL (from Prem) ---")
            print(sql)
        else:
            print("\n--- DETERMINISTIC SQL (no Prem) ---")
            print(reasoning)

