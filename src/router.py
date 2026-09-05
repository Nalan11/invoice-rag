"""Query intent router: classifies incoming user queries as 'sql' or 'rag' with SQL as primary default."""

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
    
    Defaults to 'sql' for structured database questions, routing to 'rag'
    for fuzzy, conceptual, or semantic exploratory queries.
    
    Returns:
        'sql': Default path for structured SQL queries, aggregations, and exact lookups.
        'rag': Semantic, exploratory, or fuzzy conceptual queries.
    """
    q = query.strip().lower()

    # Rule 1: Explicit Invoice ID pattern (e.g., INV-2026-0145) -> SQL
    if re.search(r'INV-\d{4}-\d{4}', query, re.IGNORECASE):
        return "sql"

    # Rule 2: Explicit Semantic / Conceptual RAG patterns
    semantic_rag_patterns = [
        "vaguely remember", "remember an invoice",
        "related to", "relating to", "concept of", "thematic",
        "employee wellbeing", "workplace comfort", "comfort", "wellbeing",
        "might cover", "similar to", "summarize vendor diversity",
        "general overview", "tell me everything about"
    ]
    if _matches_any_pattern(q, semantic_rag_patterns):
        return "rag"

    # Rule 3: Aggregation, Math, Pricing, and Ranking operations -> SQL
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

    # Rule 4: Structured lookup / specific schema attribute queries -> SQL
    lookup_patterns = [
        "billing date", "issue date", "date of invoice", "payment terms",
        "vendor issued", "which vendor", "who issued"
    ]
    if _matches_any_pattern(q, lookup_patterns):
        return "sql"

    # Rule 5: Temporal / date range filtering -> SQL
    date_patterns = [
        "last month", "this month", "last week", "this week",
        r'\blast\s+\d+\s+days\b', r'\bin\s+20\d\d\b', r'\bq[1-4]\s*20\d\d\b', r'\bq[1-4]\b',
        "between", "month-over-month",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    if _matches_any_pattern(q, date_patterns):
        return "sql"

    # Rule 6: Structured list / vendor filtering -> SQL
    filter_patterns = [
        "all invoices from", "invoices issued by", "from vendor", "list all",
        "all invoices", "show me all invoices", "list every invoice",
        "financial summary", "financial report"
    ]
    if _matches_any_pattern(q, filter_patterns):
        return "sql"

    # Default: SQL is the default path for structured invoice dataset;
    # pipeline executes RAG fallback automatically if SQL yields no results or errors.
    return "sql"
