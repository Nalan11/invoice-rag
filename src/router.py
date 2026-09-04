"""Query intent router: classifies incoming user queries as 'sql' or 'rag'."""

import re

def _matches_any_pattern(query_lower: str, patterns: list[str]) -> bool:
    """Match keywords/patterns using word boundaries to prevent false positive substring matching."""
    for pat in patterns:
        if pat.startswith(r'\b') or '(' in pat:
            if re.search(pat, query_lower, re.IGNORECASE):
                return True
        else:
            pattern = r'\b' + re.escape(pat) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
    return False

def classify_query(query: str) -> str:
    """Classify query intent using deterministic rule matching.
    
    Returns:
        'sql': Query requires structured SQL (exact ID lookup, aggregation, date range, vendor filtering).
        'rag': Query is fuzzy, semantic, or unstructured contextual search.
    """
    q = query.strip().lower()

    # Rule 1: Explicit Invoice ID pattern (e.g., INV-2026-0145) - Highest Confidence Signal
    if re.search(r'INV-\d{4}-\d{4}', query, re.IGNORECASE):
        return "sql"

    # Rule 2: Aggregation, Math, Pricing, and Ranking operations (using word boundaries & specific phrases)
    aggregation_patterns = [
        "total spent", "total sales", "total number", "total amount", "total cost",
        "total revenue", "total tax", "total quantity", "in total", "grand total", "subtotal",
        "sum of", "how many", "how much", "count of", "number of",
        "average", "avg", "highest", "lowest", "most expensive", "cheapest",
        r'\btop\s+\d+\b', r'\bbottom\s+\d+\b', "rank", "ranking",
        "unit price", "unit cost", "price of", "cost of", "quantity", "qty", "amount spent",
        "who bought", "who ordered", "who purchased", "which buyer", "buyers have purchased",
        "combined grand total", "distinct", "exceeds"
    ]
    if _matches_any_pattern(q, aggregation_patterns):
        return "sql"

    # Rule 3: Structured lookup / specific schema attribute queries
    lookup_patterns = [
        "billing date", "issue date", "date of invoice", "payment terms",
        "vendor issued", "which vendor", "who issued"
    ]
    if _matches_any_pattern(q, lookup_patterns):
        return "sql"

    # Rule 4: Temporal / date range filtering
    date_patterns = [
        "last month", "this month", "last week", "this week",
        r'\blast\s+\d+\s+days\b', r'\bin\s+20\d\d\b', r'\bq[1-4]\s*20\d\d\b', r'\bq[1-4]\b',
        "between", "month-over-month",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    if _matches_any_pattern(q, date_patterns):
        return "sql"

    # Rule 5: Structured list / vendor filtering
    filter_patterns = [
        "all invoices from", "invoices issued by", "from vendor", "list all",
        "all invoices", "show me all invoices", "list every invoice",
        "financial summary", "financial report"
    ]
    if _matches_any_pattern(q, filter_patterns):
        return "sql"

    # Default fallback: Fuzzy semantic RAG search
    return "rag"
