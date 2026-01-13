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
{
  "intent": "ONE OF: CREATE_TABLE | SELECT | INSERT | UPDATE | DELETE | ALTER | DROP",
  "table": "string (lowercase, singular)",
  "set": {},
  "where": {},
  "followup": false
}
"""

def build_prompt(user_request: str) -> str:
    return f"""
You are a STRICT SQL intent classifier and extractor.

Rules:
- Output EXACTLY ONE JSON object
- Do NOT use markdown
- Do NOT explain
- Do NOT repeat JSON
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

if __name__ == "__main__":
    tests = [
        "Increase salary to 70000 for employee with id 5",
        "Show all customers from India",
        "Create a table to store orders with date and amount",
        "Delete user where email = test@gmail.com",
        "Add a new product with name phone and price 500",
        "Update order status to shipped where order id is 10"
    ]

    for t in tests:
        print("\n" + "=" * 80)
        print("USER:", t)
        run_reasoning(t)