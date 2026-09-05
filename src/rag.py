"""RAG pipeline: context retrieval and grounded LLM answer generation."""

from src.config import get_llm_client, LLM_MODEL, llm_chat_completion
from src.search import hybrid_search

RAG_SYSTEM_PROMPT = (
    "You are InvoiceInsight, an expert financial assistant.\n"
    "Answer the user's question clearly and concisely using ONLY the provided invoice context.\n"
    "If multiple matching invoices exist in the context, list all of them with their invoice IDs, dates, and relevant details.\n"
    "If the information is not present in the context, say 'I could not find this in the available invoices.'\n"
    "Always cite the invoice ID in your response.\n"
    "SCOPE LIMITATION: Note that the provided context only contains the top retrieved documents. "
    "Do not claim absolute completeness across the entire database or extrapolate beyond the retrieved context."
)

def generate_answer(query: str, retrieved_chunks: list[dict], client=None, llm_model: str = None, **kwargs) -> str:
    """Send retrieved context and user query to LLM endpoint using the unified grounded RAG prompt."""
    if client is None:
        client = get_llm_client()

    context_blocks = [f"--- Document {i+1} ---\n{chunk['content_text']}" for i, chunk in enumerate(retrieved_chunks)]
    context_str = "\n\n".join(context_blocks)

    response = llm_chat_completion(
        client=client,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Invoice Context:\n{context_str}\n\nQuestion: {query}"}
        ],
        model=llm_model or LLM_MODEL,
        temperature=0,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def rag_answer(query: str, model, db_conn, top_k: int = 5, client=None, llm_model: str = None, **kwargs) -> dict:
    """Execute complete RAG flow: Retrieve Context -> Generate Grounded LLM Answer."""
    chunks = hybrid_search(query, model, db_conn, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "I could not find any relevant invoices in the database.",
            "retrieved_chunks": [],
            "method": "rag"
        }

    try:
        answer = generate_answer(query, chunks, client=client, llm_model=llm_model)
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            answer = "⚠️ The AI service is currently busy (rate limit reached). Please wait a moment and try again."
        else:
            answer = "⚠️ Unable to generate response from AI service. Please try again."

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "method": "rag"
    }
