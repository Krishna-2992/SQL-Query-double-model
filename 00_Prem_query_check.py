import requests


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

# print(run_prem_sql("Max and min price in electronics category"))

query = """
SQL prompt

You are an expert SQL generator.

Input is a JSON specification of a SQL query.
Generate the most accurate SQL query possible.

Rules:
- Use standard ANSI SQL
- Do not hallucinate tables or columns
- Do not explain anything
- Output only SQL

JSON:
{
  "intent": "SELECT",
  "tables": [
    "products"
  ],
  "columns": [
    "name",
    "price"
  ],
  "joins": [],
  "filters": [
    {
      "column": "name",
      "operator": "LIKE",
      "value": "%Pro%",
      "logical": "AND"
    }
  ],
  "group_by": null,
  "having": null,
  "order_by": [
    {
      "column": "price",
      "direction": "ASC"
    }
  ],
  "limit": null,
  "offset": null,
  "set": null,
  "followup": false
}

SQL:
"""
print(run_prem_sql(query))
