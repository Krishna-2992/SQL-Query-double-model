EVAL_CASES = [
    {
        "input": "Increase salary to 70000 for employee with id 5",
        "expected": {
            "intent": "UPDATE",
            "table": "employee",
            "columns": {"salary": 70000},
            "condition": "id = 5"
        }
    },
    {
        "input": "Show all customers from India",
        "expected": {
            "intent": "SELECT",
            "table": "customers"
        }
    },
    {
        "input": "Create a table to store orders with date and amount",
        "expected": {
            "intent": "CREATE_TABLE",
            "table": "orders"
        }
    },
    {
        "input": "Delete user where email = test@gmail.com",
        "expected": {
            "intent": "DELETE",
            "table": "user",
            "condition": "email = test@gmail.com"
        }
    }
]