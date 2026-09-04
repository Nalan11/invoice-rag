"""Batch execution script to test RAG system across 20 creative, multi-level difficulty queries."""

import time
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.db import get_db_connection
from src.pipeline import answer_query
from src.feedback import log_query_telemetry

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 20 Creative, Unique, Multi-Level Difficulty Test Queries
TEST_QUERIES = [
    # Level 1: Direct SQL ID Lookups (Structured)
    "What is the billing date for invoice INV-2026-0145?",
    "What is the grand total for invoice INV-2025-0033?",
    "What was the subtotal amount on invoice INV-2026-0176?",
    "Which vendor issued invoice INV-2026-0134?",
    "What are the payment terms for INV-2025-0002?",
    
    # Level 2: Aggregation & Pattern SQL Lookups
    "How many Server Rack items were purchased in invoice INV-2026-0155?",
    "What was the unit price of Extension Cord on invoice INV-2025-0088?",
    "Show me all invoices issued by Elite Cleaners.",
    "What was the total amount spent on Laptops last month?",
    "Find the total tax amount across invoices billed in May 2026.",
    
    # Level 3: Fuzzy Document & Vendor Discovery (RAG Path)
    "Find the invoice document where we purchased Laptop.",
    "I remember placing an order with PrintPro Solutions, can you locate the matching invoice record?",
    "Which invoice record covers supplies delivered to Johnson LLC?",
    "Were there any orders involving UPS Backup Units?",
    "Can you track down the invoice for ergonomic office furniture or chairs?",
    
    # Level 4: Complex Multi-Constraint & Edge Case Queries
    "Locate any invoice mentioning cleaning products or maintenance services from March 2026.",
    "Which vendor provided hardware peripherals with payment terms of Net 30?",
    "Find the receipt for network cables or wiring equipment.",
    "Can you identify invoices that incurred shipping fees or express delivery charges?",
    "What is the invoice number for our largest single purchase of desktop computers?"
]

def main():
    print("==================================================================")
    print("   RUNNING BATCH RAG TEST (20 CREATIVE MULTI-LEVEL QUERIES)      ")
    print("==================================================================")
    
    print("\nLoading SentenceTransformer embedding model and database connection...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = get_db_connection()
    
    results = []
    
    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/20] Processing Query: '{query}'")
        start_time = time.time()
        
        try:
            res = answer_query(query, model, conn)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            answer_text = res.get("answer", "No response.")
            method = res.get("method", "rag")
            
            if method == "sql":
                retrieved_ids = [res.get("sql")] if res.get("sql") else []
            else:
                chunks = res.get("retrieved_chunks") or []
                retrieved_ids = [c.get("invoice_id") for c in chunks if c.get("invoice_id")]
            
            # Log into PostgreSQL telemetry table so it shows up in history & monitoring dashboard
            row_id = log_query_telemetry(conn, query, answer_text, method, retrieved_ids, elapsed_ms, user_rating=0)
            
            print(f" -> Method   : [{method.upper()}]")
            print(f" -> Latency  : {elapsed_ms} ms")
            print(f" -> Answer   : {answer_text[:120]}...")
            print(f" -> Logged ID: #{row_id}")
            
            results.append({
                "id": idx,
                "query": query,
                "method": method,
                "latency_ms": elapsed_ms,
                "status": "SUCCESS",
                "answer_preview": answer_text[:80]
            })
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            print(f" -> ERROR    : {str(e)}")
            results.append({
                "id": idx,
                "query": query,
                "method": "ERROR",
                "latency_ms": elapsed_ms,
                "status": f"FAILED ({str(e)[:40]})",
                "answer_preview": "N/A"
            })
        
        # 1.5 second delay to avoid hitting LLM API rate limits
        time.sleep(1.5)
        
    conn.close()
    
    # Print Summary Table
    df_res = pd.DataFrame(results)
    print("\n==================================================================")
    print("                    BATCH TEST SUMMARY RESULTS                   ")
    print("==================================================================")
    print(df_res[["id", "method", "latency_ms", "status", "query"]].to_string(index=False))

if __name__ == "__main__":
    main()
