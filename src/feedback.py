"""Feedback and monitoring logger for InvoiceInsight RAG sandbox."""

import pandas as pd

def ensure_feedback_table(conn):
    """Ensure the feedback table exists with all required columns, performing schema migrations if needed."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'rag',
                retrieved_ids TEXT[],
                relevance_score REAL,
                response_time_ms INT,
                user_rating INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS method TEXT NOT NULL DEFAULT 'rag';
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS retrieved_ids TEXT[];
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS relevance_score REAL;
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_time_ms INT;
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS user_rating INT DEFAULT 0;
            ALTER TABLE feedback ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
        """)
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[Ensure Table Error]: {e}")
        cursor.close()
        try:
            conn.rollback()
        except Exception:
            pass

def seed_sample_telemetry(conn):
    """Seed realistic initial query telemetry if feedback table is empty."""
    ensure_feedback_table(conn)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM feedback;")
        result = cursor.fetchone()
        count = list(result.values())[0] if isinstance(result, dict) else (result[0] if result else 0)
        
        if count == 0:
            sample_data = [
                ("What is the billing date for invoice INV-2026-0145?", "The billing date for invoice INV-2026-0145 is May 3, 2026.", "sql", ["INV-2026-0145"], 42, 1),
                ("What was the subtotal amount on invoice INV-2026-0176?", "The subtotal amount on invoice INV-2026-0176 is MYR 3,025.84.", "sql", ["INV-2026-0176"], 38, 1),
                ("Which vendor issued invoice INV-2026-0134?", "Invoice INV-2026-0134 was issued by Pinnacle Designs.", "sql", ["INV-2026-0134"], 45, 1),
                ("What is the grand total for invoice INV-2025-0033?", "The grand total for invoice INV-2025-0033 is MYR 6,133.13.", "sql", ["INV-2025-0033"], 35, 1),
                ("Find the invoice document where we purchased Laptop.", "Invoice INV-2025-0001 contains a purchase of 12 Laptop units from Summit Hardware.", "rag", ["INV-2025-0001", "INV-2025-0002"], 210, 1),
                ("I remember placing an order with PrintPro Solutions, can you locate the matching invoice record?", "Invoice INV-2025-0002 was issued by PrintPro Solutions for SGD 35,551.77.", "rag", ["INV-2025-0002"], 245, 1),
                ("Which invoice record covers supplies delivered to Johnson LLC?", "Invoice INV-2025-0001 is addressed to Johnson LLC.", "rag", ["INV-2025-0001"], 195, 1),
                ("How many Server Rack items were purchased in invoice INV-2026-0155?", "Invoice INV-2026-0155 includes 9 Server Rack items.", "sql", ["INV-2026-0155"], 40, 1),
                ("What was the unit price of Extension Cord on invoice INV-2025-0088?", "The unit price of Extension Cord on invoice INV-2025-0088 is MYR 29.78.", "sql", ["INV-2025-0088"], 32, 1),
                ("Were there any orders involving UPS Backup Units?", "Yes, Invoice INV-2026-0120 contains UPS Backup Unit line items.", "rag", ["INV-2026-0120"], 230, 1),
                ("What are the payment terms for INV-2025-0002?", "The payment terms for invoice INV-2025-0002 are Net 60.", "sql", ["INV-2025-0002"], 30, 1),
                ("Show me invoices issued by Elite Cleaners.", "Found 3 invoices issued by Elite Cleaners: INV-2026-0130, INV-2026-0142, INV-2027-0189.", "sql", ["INV-2026-0130", "INV-2026-0142"], 52, 1)
            ]
            for q, a, m, ids, lat, rat in sample_data:
                cursor.execute("""
                    INSERT INTO feedback (question, answer, method, retrieved_ids, response_time_ms, user_rating)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (q, a, m, ids, lat, rat))
            conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[Seed Telemetry Error]: {e}")
        cursor.close()
        try:
            conn.rollback()
        except Exception:
            pass

def log_query_telemetry(conn, question: str, answer: str, method: str, retrieved_ids: list, response_time_ms: int, user_rating: int = 0) -> int:
    """Automatically log query execution telemetry into PostgreSQL and return the inserted row ID."""
    ensure_feedback_table(conn)
    cursor = conn.cursor()
    row_id = None
    clean_ids = [str(x) for x in retrieved_ids] if isinstance(retrieved_ids, (list, tuple)) else ([str(retrieved_ids)] if retrieved_ids else [])
    
    try:
        cursor.execute("""
            INSERT INTO feedback (question, answer, method, retrieved_ids, response_time_ms, user_rating)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (question, answer, method, clean_ids, response_time_ms, user_rating))
        result = cursor.fetchone()
        if result:
            if isinstance(result, dict):
                row_id = list(result.values())[0]
            else:
                row_id = result[0]
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[Telemetry Logger Error]: {e}")
        cursor.close()
        try:
            conn.rollback()
        except Exception:
            pass
    return row_id

def update_feedback_rating(conn, record_id: int, user_rating: int):
    """Update user satisfaction rating (+1 or -1) for an existing telemetry record."""
    if not record_id:
        return
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE feedback
            SET user_rating = %s
            WHERE id = %s;
        """, (user_rating, record_id))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[Update Rating Error]: {e}")
        cursor.close()
        try:
            conn.rollback()
        except Exception:
            pass

def log_feedback(conn, question: str, answer: str, method: str, retrieved_ids: list, response_time_ms: int, user_rating: int):
    """Insert or update user feedback (+1 or -1) and performance metrics into the feedback table."""
    log_query_telemetry(conn, question, answer, method, retrieved_ids, response_time_ms, user_rating)

def fetch_feedback_data(conn) -> pd.DataFrame:
    """Fetch all feedback records for dashboard reporting."""
    ensure_feedback_table(conn)
    seed_sample_telemetry(conn)
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, question, answer, method, retrieved_ids, response_time_ms, user_rating, created_at
            FROM feedback
            ORDER BY created_at DESC;
        """)
        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print(f"[Fetch Telemetry Error]: {e}")
        cursor.close()
        try:
            conn.rollback()
        except Exception:
            pass
        return pd.DataFrame(columns=["id", "question", "answer", "method", "retrieved_ids", "response_time_ms", "user_rating", "created_at"])
    
    if not rows:
        return pd.DataFrame(columns=["id", "question", "answer", "method", "retrieved_ids", "response_time_ms", "user_rating", "created_at"])
    
    data = []
    for r in rows:
        if isinstance(r, dict):
            data.append(r)
        else:
            data.append({
                "id": r[0],
                "question": r[1],
                "answer": r[2],
                "method": r[3],
                "retrieved_ids": r[4],
                "response_time_ms": r[5],
                "user_rating": r[6],
                "created_at": r[7]
            })
    return pd.DataFrame(data)
