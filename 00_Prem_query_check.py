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
print(run_prem_sql("List all employees and their department names"))
