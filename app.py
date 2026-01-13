import torch
import json
import re
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from sql_validations import validate_sql

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
- tables: list of tables(string format) involved
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


def remove_empty_fields(obj):
    """
    Recursively remove:
    - None
    - empty lists []
    - empty dicts {}
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = remove_empty_fields(v)
            if v is None:
                continue
            if isinstance(v, (list, dict)) and not v:
                continue
            cleaned[k] = v
        return cleaned

    elif isinstance(obj, list):
        cleaned_list = []
        for item in obj:
            item = remove_empty_fields(item)
            if item is None:
                continue
            if isinstance(item, (list, dict)) and not item:
                continue
            cleaned_list.append(item)
        return cleaned_list

    return obj

def normalize_reasoning(result: dict) -> dict:
    if not result:
        return None

    # ---- intent normalization ----
    intent_map = {
        "select": "SELECT",
        "retrieve": "SELECT",
        "get": "SELECT",
        "insert": "INSERT",
        "add": "INSERT",
        "update": "UPDATE",
        "delete": "DELETE",
        "remove": "DELETE",
        "create_table": "CREATE_TABLE",
    }

    raw_intent = result.get("intent", "")
    raw_intent = raw_intent.lower().strip()
    result["intent"] = intent_map.get(raw_intent, raw_intent.upper())

    # ---- normalize tables ----
    if isinstance(result.get("tables"), list):
        result["tables"] = [t.lower() for t in result["tables"]]

    # ✅ REMOVE empty / null fields
    result = remove_empty_fields(result)

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

    # print("\n--- RAW OUTPUT ---")
    # print(decoded)

    parsed = extract_json(decoded)

    # print("🎲🎲🎲🎲🎲🎲🎲🎲🎲🎲")
    # print(parsed)
    # print("🎲🎲🎲🎲🎲🎲🎲🎲🎲🎲")
    parsed = normalize_reasoning(parsed)
    # print("🛟🛟🛟🛟🛟🛟🛟🛟")
    # print(parsed)
    # print("🛟🛟🛟🛟🛟🛟🛟🛟")

    required_fields = ["intent", "tables"]
    if not parsed or not all(k in parsed and parsed[k] for k in required_fields):
        print("❌ Insufficient information to generate SQL")
        return None
        
    # print("\n--- PARSED JSON ---")
    if parsed:
        # print(json.dumps(parsed, indent=2))
        pass
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

def build_insert_sql(reasoning):
    table = reasoning["tables"][0]
    cols = reasoning["set"]["columns"]
    vals = reasoning["set"]["values"]

    col_str = ", ".join(cols)
    val_str = ", ".join(format_value(v) for v in vals)

    return f"INSERT INTO {table} ({col_str}) VALUES ({val_str});"


if __name__ == "__main__":
    # tests = [
    #     "Increase salary to 70000 for employee with id 5",
    #     "Show all customers from India",
    #     "Create a table to store orders with date and amount",
    #     "Delete user where email = test@gmail.com",
    #     "Add a new product with name phone and price 500",
    #     "Update order status to shipped where order id is 10", 
    #     "Show top 5 customers by total order value in 2023",
    #     "schema: User {id, name}, Order {order_id, price, user_id}, question: which user purchased things of more than 1000 rs."
    # ]

    # tests = [
    #     # --- LEVEL 1: Simple CRUD & Basic Filtering ---
    #     "Show all columns for the users table",
    #     "List the names of all products where the price is greater than 100",
    #     "Find the email of the user with id 50",
    #     "Add a new employee named 'Alice' with a salary of 65000 to the staff table",
    #     "Delete all records from the logs table where the level is 'DEBUG'",
    #     "Update the stock_count to 0 for product_id 202",

    #     # --- LEVEL 2: Logical Operators, Ordering, & Limits ---
    #     "Show the 10 most expensive items in the inventory",
    #     "Get all orders from '2023-05-01' to '2023-05-31' that are still 'Pending'",
    #     "List all customers from 'London' or 'Paris' who signed up in 2024",
    #     "Find all products whose name contains the word 'Pro' and order them by price ascending",

    #     # --- LEVEL 3: Aggregations & Grouping ---
    #     "What is the average salary of employees in the 'Engineering' department?",
    #     "Count the total number of orders placed by each customer",
    #     "Show the total revenue grouped by product category",
    #     "List departments that have a total salary expense exceeding 500,000",
    #     "Find the maximum and minimum price in the electronics category",

    #     # --- LEVEL 4: Joins & Relational Reasoning ---
    #     "Show the names of customers and the dates of their orders by joining customers and orders",
    #     "List all employees and their department names",
    #     "Show all products and their respective supplier names where the supplier is based in 'USA'",
    #     "Get a list of all students and the names of the courses they are enrolled in",
    #     "Find all users who bought a product in the 'Home Decor' category",

    #     # --- LEVEL 5: Complex Reasoning & Subqueries ---
    #     "Show the top 5 customers by total order value in 2023",
    #     "Find all products with a price higher than the average price of all products",
    #     "List the names of users who have never placed an order",
    #     "For each city, show the total number of users and the average order value",
    #     "Find the employee who has the highest salary in the 'Marketing' department",
        
    #     # --- LEVEL 6: Schema & Constraints (DDL) ---
    #     "Create a table named 'tasks' with id, title, and a boolean completed field",
    #     "Add a new column 'last_login' of type timestamp to the users table"
    # ]

    tests = [
        # "Show all columns for the users table",
        # "Insert a staff record",
        # "Find all products whose name contains the word 'Pro' and order them by price ascending",
        "Count the total number of orders placed by each customer",
        # "List departments that have a total salary expense exceeding 500,000",
        # "Find the maximum and minimum price in the electronics category",
        # "Show all products and their respective supplier names where the supplier is based in 'USA'",
        # "Get a list of all students and the names of the courses they are enrolled in",
        # "List the names of users who have never placed an order",
        # "For each city, show the total number of users and the average order value"
    ]

    for t in tests:
        # print("\n" + "=" * 80)
        # print("USER:", t)

        reasoning = run_reasoning(t)



        if not reasoning or reasoning["intent"] == "UNKNOWN":
            print("❌ Could not reason about query")
            continue

        print("🚀reasoning🚀🚀🚀🚀🚀")
        print(reasoning)
        print("🚀🚀🚀🚀🚀🚀")
        
        # ---------- INSERTING LOGIC --------
        # if reasoning["intent"] == "INSERT": 
        #     print("inside insert intent block")
        #     sql = build_insert_sql(reasoning)
        #     print("user query: ", t)
        #     print(sql)
        #     continue

        sql_prompt = build_sql_prompt_from_reasoning(reasoning)

        print("`🏀🏀🏀🏀🏀🏀🏀🏀🏀SQL prompt")
        print(sql_prompt)
        print("🏀`🏀🏀🏀🏀🏀🏀🏀")

        if sql_prompt:
            sql = run_sql_generation(sql_prompt)

            # print("\n--- GENERATED SQL (from Prem) ---")
            # print(sql)

            SCHEMA = {}
            is_valid, validation_errors = validate_sql(
                sql=sql,
                reasoning=reasoning,
                schema=SCHEMA
            )

            if not is_valid:
                print("❌ SQL VALIDATION FAILED")
                print("Reasons:")
                for err in validation_errors:
                    print(f"  - {err}")
                continue

            # print("✅ SQL VALIDATION PASSED")
            # print("\n--- FINAL SAFE SQL ---")
            print("user query: ", t)
            print(sql)

        else:
            print("\n--- DETERMINISTIC SQL (no Prem) ---")
            print(reasoning)

