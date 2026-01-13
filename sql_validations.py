from sqlglot import parse_one, errors, exp

def validate_sql_syntax(sql: str) -> bool:
    try:
        parse_one(sql)
        return True
    except errors.ParseError:
        return False

def validate_intent_sql_match(intent: str, sql: str) -> bool:
    sql_upper = sql.strip().upper()
    return sql_upper.startswith(intent)



def extract_tables(sql: str) -> set:
    ast = parse_one(sql)
    return {t.name.lower() for t in ast.find_all(exp.Table)}

def validate_tables(sql: str, allowed_tables: list) -> bool:
    sql_tables = extract_tables(sql)
    allowed = {t.lower() for t in allowed_tables}
    return sql_tables.issubset(allowed)



def extract_columns(sql: str) -> set:
    ast = parse_one(sql)
    return {c.name.lower() for c in ast.find_all(exp.Column)}

def validate_columns(sql: str, schema: dict):
    if not schema:
        return True, []

    columns = extract_columns(sql)

    valid_columns = {
        c.lower()
        for cols in schema.values()
        for c in cols
    }

    invalid = columns - valid_columns

    if invalid:
        return False, [
            f"Invalid columns used: {invalid}. "
            f"Valid columns: {valid_columns}"
        ]

    return True, []
        

def has_aggregate(sql: str) -> bool:
    return any(fn in sql.upper() for fn in ["SUM(", "COUNT(", "AVG(", "MIN(", "MAX("])

def validate_group_by(sql: str) -> bool:
    if has_aggregate(sql):
        return "GROUP BY" in sql.upper()
    return True

def validate_limit(sql: str, reasoning: dict) -> bool:
    if reasoning.get("limit") and "LIMIT" not in sql.upper():
        return False
    return True

def validate_safe_where(sql: str, intent: str) -> bool:
    if intent in ["UPDATE", "DELETE"]:
        return "WHERE" in sql.upper()
    return True

# FORBIDDEN = [
#     "DROP TABLE",
#     "TRUNCATE",
#     "ALTER TABLE",
#     "--",
#     ";--",
#     "/*",
#     "*/"
# ]

# def validate_safety(sql: str) -> bool:
#     sql_upper = sql.upper()
#     return not any(f in sql_upper for f in FORBIDDEN)

def validate_sql(sql: str, reasoning: dict, schema: dict):
    errors = []

    # 1. Syntax
    if not validate_sql_syntax(sql):
        errors.append("SQL syntax is invalid")

    # 2. Intent
    if not validate_intent_sql_match(reasoning["intent"], sql):
        errors.append(
            f"SQL intent mismatch: expected {reasoning['intent']}"
        )

    # 3. Tables
    try:
        if not validate_tables(sql, reasoning["tables"]):
            sql_tables = extract_tables(sql)
            errors.append(
                f"SQL uses tables {sql_tables}, "
                f"but only {reasoning['tables']} are allowed"
            )
    except Exception as e:
        errors.append(f"Failed to extract tables: {str(e)}")

    # 4. Columns
    try:
        if not validate_columns(sql, schema):
            sql_columns = extract_columns(sql)
            valid_columns = {
                c.lower()
                for cols in schema.values()
                for c in cols
            }
            errors.append(
                f"SQL uses columns {sql_columns}, "
                f"but valid columns are {valid_columns}"
            )
    except Exception as e:
        errors.append(f"Failed to extract columns: {str(e)}")

    # 5. GROUP BY
    if not validate_group_by(sql):
        errors.append(
            "Aggregate function used without GROUP BY clause"
        )

    # 6. Unsafe UPDATE / DELETE
    if not validate_safe_where(sql, reasoning["intent"]):
        errors.append(
            f"{reasoning['intent']} query without WHERE clause is unsafe"
        )

    # Final result
    return len(errors) == 0, errors