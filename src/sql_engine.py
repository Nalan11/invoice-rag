"""Text-to-SQL execution engine for structured queries over invoice_chunks.content_json."""

import json
import re
from decimal import Decimal
from src.config import get_llm_client, LLM_MODEL, llm_chat_completion

SQL_SYSTEM_PROMPT = """You are an expert PostgreSQL data analyst.
You have a table `invoice_chunks` with a JSONB column `content_json` containing invoice data.

Column schema of `invoice_chunks`:
- invoice_id (TEXT): Unique invoice code (e.g. 'INV-2026-0145')
- content_json (JSONB): Structured JSON object with keys:
  - invoice_id (text)
  - vendor (text)
  - date (text, format YYYY-MM-DD)
  - buyer (text)
  - buyer_address (text)
  - subtotal (numeric)
  - tax_rate (text)
  - tax (numeric)
  - grand_total (numeric)
  - currency (text, e.g. 'MYR', 'USD', 'SGD')
  - payment_terms (text)
  - line_items (array of objects with keys: description, qty, unit_price, amount)

CRITICAL RULES FOR SQL GENERATION:
1. When asked for total item quantities or "who bought the most <item>", you MUST SUM the quantity `(item->>'qty')::int` from `jsonb_array_elements(content_json->'line_items')`:
   SELECT content_json->>'buyer' as buyer, SUM((item->>'qty')::int) as total_qty
   FROM invoice_chunks, jsonb_array_elements(content_json->'line_items') AS item
   WHERE item->>'description' ILIKE '%Laptop%'
   GROUP BY content_json->>'buyer'
   ORDER BY total_qty DESC;
2. Do NOT use `COUNT(*)` when the user asks for total items bought. `COUNT(*)` counts invoices, NOT item quantities! Use `SUM((item->>'qty')::int)`.
3. For compound questions (e.g. "Who bought the most laptop? How many people are there?"), query all buyers who bought the item along with their sum of quantities so the breakdown can be calculated.
4. For vendor or buyer searches, use `ILIKE '%Name%'` on `content_json->>'vendor'` or `content_json->>'buyer'`.
5. For multi-invoice ID inquiries (e.g. INV-2025-0001 and INV-2026-0192), filter using `WHERE content_json->>'invoice_id' IN ('INV-2025-0001', 'INV-2026-0192')`.
6. Write ONLY a valid, read-only PostgreSQL SELECT query. Do NOT include markdown code blocks, explanation, or extra text."""

FORBIDDEN_SQL_PATTERNS = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b',
    re.IGNORECASE
)

def validate_sql(sql: str) -> bool:
    """Validate that the generated SQL query is strictly read-only and safe."""
    stripped = sql.strip().rstrip(';').strip()
    if not (stripped.upper().startswith('SELECT') or stripped.upper().startswith('WITH')):
        return False
    if FORBIDDEN_SQL_PATTERNS.search(stripped):
        return False
    return True

def generate_sql(query: str, client=None, llm_model: str = None) -> str:
    """Generate PostgreSQL query from natural language question using LLM."""
    if client is None:
        client = get_llm_client()

    response = llm_chat_completion(
        client=client,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Write a SQL query to answer: {query}"}
        ],
        model=llm_model or LLM_MODEL,
        temperature=0,
        max_tokens=400
    )
    
    raw_sql = response.choices[0].message.content.strip()
    
    # Clean code fences if LLM included markdown
    cleaned_sql = re.sub(r'^```(?:sql)?\s*', '', raw_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'\s*```$', '', cleaned_sql).strip()
    return cleaned_sql

def text_to_sql_answer(query: str, db_conn, client=None, llm_model: str = None) -> dict:
    """Execute Text-to-SQL workflow: Question → SQL → Execution → Natural Language Answer."""
    if client is None:
        client = get_llm_client()

    id_matches = re.findall(r'INV-\d{4}-\d{4}', query, re.IGNORECASE)
    sql_params = None

    # Fast-path optimization: Single exact invoice ID lookup without complex aggregation question
    is_simple_lookup = len(id_matches) == 1 and not any(kw in query.lower() for kw in ["how many", "sum", "total", "line item", "items", "average"])
    
    if is_simple_lookup:
        target_id = id_matches[0].upper()
        sql_query = "SELECT content_json FROM invoice_chunks WHERE content_json->>'invoice_id' = %s;"
        sql_params = (target_id,)
    else:
        try:
            sql_query = generate_sql(query, client, llm_model=llm_model)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                user_msg = "⚠️ The AI service is currently busy (rate limit reached). Please wait a moment and try again."
            else:
                user_msg = "⚠️ Unable to generate database query. Please try rephrasing your question."
            return {
                "answer": user_msg,
                "sql": None,
                "data": None,
                "method": "sql"
            }

    # Security & Safety Validation
    if not sql_params and not validate_sql(sql_query):
        return {
            "answer": "Security Alert: Generated SQL query violated read-only safety rules and execution was blocked.",
            "sql": sql_query,
            "data": None,
            "method": "sql"
        }

    # Execute SQL
    cursor = db_conn.cursor()
    try:
        if sql_params:
            cursor.execute(sql_query, sql_params)
        else:
            cursor.execute(sql_query)
        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        cursor.close()
        try:
            db_conn.rollback()
        except Exception:
            pass
            
        # Fallback to direct ID match if generated SQL errored but an ID was present
        if id_matches:
            target_id = id_matches[0].upper()
            sql_query = "SELECT content_json FROM invoice_chunks WHERE content_json->>'invoice_id' = %s;"
            cursor = db_conn.cursor()
            try:
                cursor.execute(sql_query, (target_id,))
                rows = cursor.fetchall()
                cursor.close()
            except Exception as e_sub:
                cursor.close()
                try:
                    db_conn.rollback()
                except Exception:
                    pass
                return {
                    "answer": "⚠️ Database query execution encountered an error. Please try again.",
                    "sql": sql_query,
                    "data": None,
                    "method": "sql"
                }
        else:
            return {
                "answer": "⚠️ Database query execution encountered an error. Please try again.",
                "sql": sql_query,
                "data": None,
                "method": "sql"
            }

    if not rows:
        return {
            "answer": "No records found matching your query criteria.",
            "sql": sql_query,
            "data": [],
            "method": "sql"
        }

    # Clean & Format DB Rows for LLM Synthesis
    try:
        results_data = []
        for r in rows:
            if isinstance(r, dict):
                clean_dict = {}
                for k, v in r.items():
                    if isinstance(v, Decimal):
                        clean_dict[k] = float(v)
                    elif isinstance(v, dict):
                        clean_dict[k] = v
                    else:
                        clean_dict[k] = v
                results_data.append(clean_dict)
            elif len(r) == 1 and isinstance(r[0], (dict, list)):
                results_data.append(r[0])
            else:
                clean_row = []
                for val in r:
                    if isinstance(val, Decimal):
                        clean_row.append(float(val))
                    else:
                        clean_row.append(val)
                results_data.append(clean_row)

        data_str = json.dumps(results_data[:25], default=str, indent=2)

        resp = llm_chat_completion(
            client=client,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are InvoiceInsight, an expert financial assistant.\n"
                        "Synthesize a clear, professional, human-readable natural language answer using the query execution data.\n"
                        "Format monetary values nicely with currency codes (e.g. MYR 14,271.66).\n"
                        "If multiple items exist, format as bullet points or a clean structured list.\n"
                        "CRITICAL: Do NOT output raw python dicts, objects, or Decimal strings under any circumstances."
                    )
                },
                {"role": "user", "content": f"User Question: {query}\n\nSQL Execution Payload:\n{data_str}"}
            ],
            model=llm_model or LLM_MODEL,
            temperature=0,
            max_tokens=1000
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            answer = "⚠️ The AI service is currently busy (rate limit reached). Please wait a moment and try again."
        elif results_data and isinstance(results_data[0], dict):
            formatted_items = []
            for item in results_data[:5]:
                formatted_items.append(", ".join([f"{k}: {v}" for k, v in item.items()]))
            answer = f"Found {len(rows)} matching record(s):\n- " + "\n- ".join(formatted_items)
        else:
            answer = f"Found {len(rows)} matching record(s)."

    return {
        "answer": answer,
        "sql": sql_query,
        "data": results_data,
        "method": "sql"
    }
