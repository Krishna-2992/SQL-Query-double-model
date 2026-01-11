import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model.eval()

REASONING_SCHEMA = """
{
  "intent": "",
  "table": "",
  "columns": {},
  "values": {},
  "condition": "",
  "followup": false
}
"""

def build_prompt(user_request: str) -> str:
    return f"""
You are a SQL reasoning engine.

Extract intent and structure from the user request.
Output ONLY valid JSON.
Do NOT explain.

JSON format:
{REASONING_SCHEMA}

User request:
{user_request}
"""

def extract_json(text: str):
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def normalize_reasoning(result: dict) -> dict:
    if not result:
        return result

    if not result.get("condition") and isinstance(result.get("values"), dict):
        if len(result["values"]) == 1:
            k, v = next(iter(result["values"].items()))
            result["condition"] = f"{k} = {v}"

    result.setdefault("columns", {})
    result.setdefault("values", {})
    result.setdefault("followup", False)

    return result

def run_reasoning(user_request: str):
    prompt = build_prompt(user_request)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- RAW OUTPUT ---")
    print(decoded)

    parsed = extract_json(decoded)
    parsed = normalize_reasoning(parsed)

    print("\n--- PARSED JSON ---")
    print(json.dumps(parsed, indent=2))

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