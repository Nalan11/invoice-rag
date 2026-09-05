"""Inspect the last 15 conversation records in detail."""

import os
import sys

# Ensure repository root is in sys.path regardless of execution directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.db import get_db_connection

conn = get_db_connection()
cur = conn.cursor()
cur.execute("""
    SELECT id, created_at, method, user_rating, response_time_ms, question, answer, retrieved_ids 
    FROM feedback 
    ORDER BY id DESC 
    LIMIT 15
""")
rows = cur.fetchall()

print(f"Fetched {len(rows)} latest records from feedback table:\n")
for r in reversed(rows):
    print("=" * 80)
    print(f"Record ID: #{r['id']} | Timestamp: {r['created_at']} | Method: [{r['method']}] | Latency: {r['response_time_ms']}ms")
    print(f"User Question : {r['question']}")
    print(f"System Answer :\n{r['answer']}")
    print(f"Context/SQL   : {r['retrieved_ids']}")

conn.close()
