"""CLI Script to inspect web conversation history and telemetry logs stored in PostgreSQL."""

import os
import sys

# Ensure repository root is in sys.path regardless of execution directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


import sys
import pandas as pd
from src.db import get_db_connection
from src.feedback import fetch_feedback_data

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("Connecting to PostgreSQL database...")
    conn = get_db_connection()
    df = fetch_feedback_data(conn)
    conn.close()

    if df.empty:
        print("No query telemetry logs found in the database yet.")
        return

    print(f"\n=======================================================")
    print(f"   INVOICEINSIGHT WEB CONVERSATION AUDIT LOG ({len(df)} Queries)")
    print(f"=======================================================\n")

    for idx, row in df.iterrows():
        rating = "+1 (Positive)" if row['user_rating'] == 1 else ("-1 (Negative)" if row['user_rating'] == -1 else "0 (Unrated)")
        method_badge = f"[{str(row['method']).upper()}]"
        
        print(f"Query #{row['id']} | Date: {row['created_at']} | Latency: {row['response_time_ms']}ms | Method: {method_badge} | Rating: {rating}")
        print(f"Question : {row['question']}")
        print(f"Answer   : {row['answer']}")
        print(f"Retrieved: {row['retrieved_ids']}")
        print("-" * 75)

if __name__ == "__main__":
    main()
