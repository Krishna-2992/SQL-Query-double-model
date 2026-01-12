import json


text = """
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


`User request`:
schema: User {id, name}, Order {order_id, price, user_id}, question: which user purchased things of more than 1000 rs.

JSON:
{
  "intent": "SELECT",
  "tables": ["User", "Order"],
  "columns": ["User.name", "sum(Order.price) as total_price"],
  "joins": [
    {
      "type": "INNER",
      "table": "Order",
      "on": "User.id = Order.user_id"
    }
  ],
  "filters": [
    {
      "column": "total_price",
      "operator": ">=",
      "value": 1000,
      "logical": "AND"
    }
  ],
  "group_by": ["User.name"],
  "having": [],
  "order_by": [],
  "limit": null,
  "offset": null,
  "set": null,
  "followup": false
}
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



print(extract_json(text))