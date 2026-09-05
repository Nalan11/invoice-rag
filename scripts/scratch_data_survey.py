"""Quick data survey to understand distribution of entities in the database."""

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

# 1. Total records
cur.execute('SELECT COUNT(*) as cnt FROM invoice_chunks')
row = cur.fetchone()
total = list(row.values())[0] if isinstance(row, dict) else row[0]
print(f"TOTAL RECORDS: {total}")

# 2. Sample vendors
cur.execute("SELECT DISTINCT content_json->>'vendor' as vendor FROM invoice_chunks ORDER BY vendor")
rows = cur.fetchall()
vendors = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
print(f"\nVENDORS ({len(vendors)}):")
for v in vendors:
    print(f"  - {v}")

# 3. Sample buyers
cur.execute("SELECT DISTINCT content_json->>'buyer' as buyer FROM invoice_chunks ORDER BY buyer")
rows = cur.fetchall()
buyers = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
print(f"\nBUYERS ({len(buyers)}):")
for b in buyers:
    print(f"  - {b}")

# 4. Line item descriptions
cur.execute("""
    SELECT DISTINCT item->>'description' as item_desc
    FROM invoice_chunks, jsonb_array_elements(content_json->'line_items') AS item
    ORDER BY item_desc
""")
rows = cur.fetchall()
items = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
print(f"\nLINE ITEM DESCRIPTIONS ({len(items)}):")
for it in items:
    print(f"  - {it}")

# 5. Currency distribution
cur.execute("SELECT content_json->>'currency' as currency, COUNT(*) as cnt FROM invoice_chunks GROUP BY currency ORDER BY cnt DESC")
rows = cur.fetchall()
print(f"\nCURRENCIES:")
for r in rows:
    vals = list(r.values()) if isinstance(r, dict) else list(r)
    print(f"  - {vals[0]}: {vals[1]} invoices")

# 6. Date range
cur.execute("SELECT MIN(content_json->>'date') as min_date, MAX(content_json->>'date') as max_date FROM invoice_chunks")
row = cur.fetchone()
vals = list(row.values()) if isinstance(row, dict) else list(row)
print(f"\nDATE RANGE: {vals[0]} to {vals[1]}")

# 7. Payment terms
cur.execute("SELECT DISTINCT content_json->>'payment_terms' as pt FROM invoice_chunks ORDER BY pt")
rows = cur.fetchall()
pts = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
print(f"\nPAYMENT TERMS ({len(pts)}):")
for pt in pts:
    print(f"  - {pt}")

# 8. Grand total stats
cur.execute("SELECT MIN((content_json->>'grand_total')::numeric), MAX((content_json->>'grand_total')::numeric), AVG((content_json->>'grand_total')::numeric) FROM invoice_chunks")
row = cur.fetchone()
vals = list(row.values()) if isinstance(row, dict) else list(row)
print(f"\nGRAND TOTAL STATS: Min={vals[0]}, Max={vals[1]}, Avg={vals[2]}")

# 9. Invoice ID pattern sample
cur.execute("SELECT invoice_id FROM invoice_chunks ORDER BY invoice_id LIMIT 10")
rows = cur.fetchall()
ids = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
print(f"\nSAMPLE INVOICE IDS: {ids}")

conn.close()
print("\n--- Survey complete ---")
