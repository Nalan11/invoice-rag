"""Unified Pipeline: Router -> (Text-to-SQL Path | RAG Path) -> Bi-Directional Fallback -> Formatted Answer."""

from src.router import classify_query
from src.sql_engine import text_to_sql_answer
from src.rag import rag_answer
from src.config import get_llm_client

def answer_query(
    query: str, 
    model, 
    db_conn, 
    client=None, 
    force_method: str = None,
    llm_model: str = None,
    **kwargs
) -> dict:
    """Execute complete unified pipeline with bi-directional fallback safety net and dynamic model selection.
    
    SQL is the default path. If SQL yields no records or errors, it falls back to RAG.
    If explicitly routed to RAG and RAG yields no confident answer, it falls back to SQL.
    
    Args:
        query: User natural language question.
        model: SentenceTransformer embedding model instance.
        db_conn: PostgreSQL connection.
        client: Optional OpenAI client instance.
        force_method: 'sql' or 'rag' to override automatic router classification.
        llm_model: Optional LLM model identifier (e.g. 'llama-3.3-70b-versatile', 'gpt-4o-mini').
        
    Returns:
        Dict containing answer, classification method, and context/SQL details.
    """
    if client is None:
        client = get_llm_client()

    method = force_method if force_method else classify_query(query)

    if method == "sql":
        result = text_to_sql_answer(query, db_conn, client=client, llm_model=llm_model)
        answer_lower = result.get("answer", "").lower()
        
        # Bi-Directional Fallback: If SQL yielded no results or errored, try RAG path
        if not result.get("data") or "no records found" in answer_lower or "error" in answer_lower:
            rag_res = rag_answer(
                query, model, db_conn, top_k=5, client=client, llm_model=llm_model
            )
            rag_answer_text = rag_res.get("answer", "").lower()
            if rag_res.get("retrieved_chunks") and "could not find" not in rag_answer_text:
                result = rag_res
                result["method"] = "rag (sql-fallback)"
    else:
        result = rag_answer(
            query, model, db_conn, top_k=5, client=client, llm_model=llm_model
        )
        answer_lower = result.get("answer", "").lower()
        
        # Bi-Directional Fallback: If RAG yielded no confident answer, try SQL path
        if not result.get("answer") or "could not find" in answer_lower:
            sql_res = text_to_sql_answer(query, db_conn, client=client, llm_model=llm_model)
            sql_answer_text = sql_res.get("answer", "").lower()
            if sql_res.get("data") and "no records found" not in sql_answer_text and "error" not in sql_answer_text:
                result = sql_res
                result["method"] = "sql (rag-fallback)"

    result["query"] = query
    result["llm_model"] = llm_model
    return result
