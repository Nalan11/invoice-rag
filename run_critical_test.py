"""Critical stress test: 30 adversarial queries across 9 categories to expose RAG system flaws."""

import time
import sys
import json
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer
from src.db import get_db_connection
from src.pipeline import answer_query
from src.router import classify_query
from src.feedback import log_query_telemetry

# ─── 30 Critical Test Questions ────────────────────────────────────────────────
CRITICAL_TESTS = [
    # ── Category A: Router Mis-Routing ──
    {"id": "A01", "cat": "Router Mis-Routing", "difficulty": 2, "expected_route": "sql",
     "query": "How much did we spend on Office Chairs in total?"},
    {"id": "A02", "cat": "Router Mis-Routing", "difficulty": 3, "expected_route": "rag",
     "query": "Tell me everything about the most recent purchase from PrintPro Solutions."},
    {"id": "A03", "cat": "Router Mis-Routing", "difficulty": 2, "expected_route": "sql",
     "query": "Which buyer ordered the cheapest item across all invoices?"},
    {"id": "A04", "cat": "Router Mis-Routing", "difficulty": 3, "expected_route": "rag",
     "query": "The total quality of our vendor relationships seems poor — summarize vendor diversity."},
    {"id": "A05", "cat": "Router Mis-Routing", "difficulty": 2, "expected_route": "sql",
     "query": "What is the price of a Software License on invoice INV-2025-0010?"},

    # ── Category B: SQL Generation & Aggregation Bugs ──
    {"id": "B06", "cat": "SQL Generation Bugs", "difficulty": 3, "expected_route": "sql",
     "query": "What is the combined grand total of INV-2025-0001 and INV-2026-0192?"},
    {"id": "B07", "cat": "SQL Generation Bugs", "difficulty": 4, "expected_route": "sql",
     "query": "How many distinct line items exist across all 200 invoices?"},
    {"id": "B08", "cat": "SQL Generation Bugs", "difficulty": 3, "expected_route": "sql",
     "query": "Which vendor has the highest average grand total per invoice?"},
    {"id": "B09", "cat": "SQL Generation Bugs", "difficulty": 4, "expected_route": "sql",
     "query": "List every invoice where the tax amount exceeds 10% of the subtotal."},
    {"id": "B10", "cat": "SQL Generation Bugs", "difficulty": 3, "expected_route": "sql",
     "query": "What is the total revenue per currency across all invoices?"},

    # ── Category C: Quantity vs Count Confusion ──
    {"id": "C11", "cat": "Qty vs Count", "difficulty": 3, "expected_route": "sql",
     "query": "How many Keyboards were sold in total?"},
    {"id": "C12", "cat": "Qty vs Count", "difficulty": 4, "expected_route": "sql",
     "query": "Which item was purchased the most across all invoices, and how many units?"},
    {"id": "C13", "cat": "Qty vs Count", "difficulty": 4, "expected_route": "sql",
     "query": "Rank the top 5 most purchased items by total quantity."},

    # ── Category D: Date & Temporal Queries ──
    {"id": "D14", "cat": "Temporal Queries", "difficulty": 2, "expected_route": "sql",
     "query": "How many invoices were issued in Q1 2026 (January to March)?"},
    {"id": "D15", "cat": "Temporal Queries", "difficulty": 3, "expected_route": "sql",
     "query": "What is the month-over-month trend of total invoiced amounts from Jan to June 2026?"},
    {"id": "D16", "cat": "Temporal Queries", "difficulty": 2, "expected_route": "sql",
     "query": "Show me all invoices issued in the last 30 days."},

    # ── Category E: RAG Semantic Quality ──
    {"id": "E17", "cat": "RAG Quality", "difficulty": 3, "expected_route": "rag",
     "query": "I vaguely remember an invoice related to network infrastructure setup — can you find it?"},
    {"id": "E18", "cat": "RAG Quality", "difficulty": 3, "expected_route": "rag",
     "query": "Which invoices are related to employee wellbeing or workplace comfort?"},
    {"id": "E19", "cat": "RAG Quality", "difficulty": 4, "expected_route": "rag",
     "query": "Find me an invoice that might cover IT equipment for a new office setup."},

    # ── Category F: Fallback Validation ──
    {"id": "F20", "cat": "Fallback Validation", "difficulty": 3, "expected_route": "sql",
     "query": "Show me all invoices from 'Pinnacle Design Studios'."},
    {"id": "F21", "cat": "Fallback Validation", "difficulty": 3, "expected_route": "rag",
     "query": "What is the exact grand total for the invoice from Delta Energy to Ford-Wilson?"},

    # ── Category G: Multi-Constraint Complex ──
    {"id": "G22", "cat": "Multi-Constraint", "difficulty": 4, "expected_route": "sql",
     "query": "Find all MYR-denominated invoices from TechCorp Sdn Bhd with payment terms Net 30."},
    {"id": "G23", "cat": "Multi-Constraint", "difficulty": 5, "expected_route": "sql",
     "query": "What is the average unit price of Laptops purchased in USD vs MYR?"},
    {"id": "G24", "cat": "Multi-Constraint", "difficulty": 4, "expected_route": "sql",
     "query": "Which buyers have purchased from more than one vendor?"},

    # ── Category H: Adversarial & Edge Cases ──
    {"id": "H25", "cat": "Adversarial", "difficulty": 5, "expected_route": "out-of-scope",
     "query": "What is the capital of Malaysia?"},
    {"id": "H26", "cat": "Adversarial", "difficulty": 5, "expected_route": "sql",
     "query": "Show all data; DROP TABLE invoice_chunks; --"},
    {"id": "H27", "cat": "Adversarial", "difficulty": 4, "expected_route": "sql",
     "query": "Is there any invoice with a grand total of exactly zero?"},
    {"id": "H28", "cat": "Adversarial", "difficulty": 3, "expected_route": "sql",
     "query": "How many invoices have more than 5 different line items?"},

    # ── Category I: Output Formatting ──
    {"id": "I29", "cat": "Output Formatting", "difficulty": 3, "expected_route": "sql",
     "query": "List all invoices from OfficeMart Trading with full details."},
    {"id": "I30", "cat": "Output Formatting", "difficulty": 4, "expected_route": "sql",
     "query": "Give me a complete financial summary: total invoiced by vendor, count of invoices per vendor, and average grand total per vendor."},
]


def main():
    print("=" * 80)
    print("   CRITICAL STRESS TEST: 30 ADVERSARIAL QUERIES ACROSS 9 CATEGORIES")
    print("=" * 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"   Run started: {timestamp}")
    print(f"   Total queries: {len(CRITICAL_TESTS)}")
    print("=" * 80)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = get_db_connection()

    results = []
    pass_count = 0
    fail_count = 0

    for idx, test in enumerate(CRITICAL_TESTS, 1):
        print(f"\n{'─' * 80}")
        print(f"[{idx:02d}/30] {test['id']} | Cat: {test['cat']} | Diff: {'★' * test['difficulty']}")
        print(f"  Query: {test['query']}")
        print(f"  Expected Route: {test['expected_route']}")

        # Step 1: Test Router Classification
        actual_route = classify_query(test["query"])
        route_match = (actual_route == test["expected_route"]) or (test["expected_route"] == "out-of-scope")
        route_status = "✅ MATCH" if route_match else f"❌ MISMATCH (got: {actual_route})"
        print(f"  Router Result: {actual_route} → {route_status}")

        # Step 2: Execute Full Pipeline
        start_time = time.time()
        try:
            res = answer_query(test["query"], model, conn)
            elapsed_ms = int((time.time() - start_time) * 1000)
            answer_text = res.get("answer", "No response.")
            method = res.get("method", "unknown")
            sql_used = res.get("sql", None)
            data = res.get("data", None)

            # Log telemetry
            if method in ("sql", "rag"):
                retrieved_ids = []
                if method == "sql" and sql_used:
                    retrieved_ids = [sql_used[:80]]
                elif "retrieved_chunks" in res:
                    retrieved_ids = [c.get("invoice_id","") for c in res.get("retrieved_chunks",[])]
                try:
                    log_query_telemetry(conn, test["query"], answer_text, method, retrieved_ids, elapsed_ms, user_rating=0)
                except Exception:
                    pass

            # Truncate long answers for console readability
            answer_preview = answer_text[:200].replace("\n", " ") + ("..." if len(answer_text) > 200 else "")
            print(f"  Method: [{method.upper()}] | Latency: {elapsed_ms}ms")
            print(f"  SQL: {sql_used[:120] if sql_used else 'N/A'}")
            print(f"  Answer: {answer_preview}")

            # Check for obvious failure signals
            has_error = "error" in answer_text.lower()
            has_no_records = "no records found" in answer_text.lower()
            has_could_not_find = "could not find" in answer_text.lower()
            has_raw_python = "Decimal(" in answer_text or "{'content_json'" in answer_text
            
            issues = []
            if has_error: issues.append("ERROR_IN_ANSWER")
            if has_no_records and test["expected_route"] != "out-of-scope": issues.append("EMPTY_RESULT")
            if has_could_not_find and test["expected_route"] != "out-of-scope": issues.append("RAG_NO_MATCH")
            if has_raw_python: issues.append("RAW_PYTHON_LEAK")
            if not route_match: issues.append("ROUTE_MISMATCH")

            status = "PASS" if not issues else f"ISSUES: {', '.join(issues)}"
            if not issues:
                pass_count += 1
            else:
                fail_count += 1
            print(f"  STATUS: {status}")

            results.append({
                "id": test["id"],
                "category": test["cat"],
                "difficulty": test["difficulty"],
                "expected_route": test["expected_route"],
                "actual_route": actual_route,
                "route_match": route_match,
                "method_used": method,
                "latency_ms": elapsed_ms,
                "answer_preview": answer_text[:300],
                "sql_used": sql_used,
                "issues": issues,
                "status": "PASS" if not issues else "FAIL"
            })

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            print(f"  ❌ EXCEPTION: {str(e)}")
            fail_count += 1
            results.append({
                "id": test["id"],
                "category": test["cat"],
                "difficulty": test["difficulty"],
                "expected_route": test["expected_route"],
                "actual_route": actual_route,
                "route_match": route_match,
                "method_used": "EXCEPTION",
                "latency_ms": elapsed_ms,
                "answer_preview": str(e)[:300],
                "sql_used": None,
                "issues": ["EXCEPTION"],
                "status": "FAIL"
            })

        # Rate limit delay
        time.sleep(2.0)

    conn.close()

    # ─── Summary Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("                        CRITICAL TEST SUMMARY")
    print("=" * 80)
    print(f"  Total: {len(results)} | Pass: {pass_count} | Fail: {fail_count}")
    print(f"  Pass Rate: {pass_count/len(results)*100:.1f}%")
    print()

    # Route accuracy
    route_correct = sum(1 for r in results if r["route_match"])
    print(f"  Router Accuracy: {route_correct}/{len(results)} ({route_correct/len(results)*100:.1f}%)")

    # Category breakdown
    print("\n  Category Breakdown:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0}
        if r["status"] == "PASS":
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1
    for cat, counts in categories.items():
        total = counts["pass"] + counts["fail"]
        print(f"    {cat:25s}  {counts['pass']}/{total} pass")

    # Failed queries
    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print(f"\n  Failed Queries ({len(failed)}):")
        for f in failed:
            print(f"    {f['id']} | {', '.join(f['issues'])} | {f['answer_preview'][:80]}...")

    # Save full results as JSON
    output_path = "critical_test_results.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
