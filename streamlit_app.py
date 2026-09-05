"""InvoiceInsight: Intelligent Invoice RAG & Text-to-SQL Assistant.

Phase 5: Streamlit Web UI + Monitoring Dashboard + Automatic Telemetry Logger.
"""

import time
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer

from src import config
from src.db import get_db_connection, get_healthy_db_connection
from src.pipeline import answer_query
from src.feedback import log_feedback, log_query_telemetry, update_feedback_rating, fetch_feedback_data, ensure_feedback_table

# Page Configuration
st.set_page_config(
    page_title="InvoiceInsight | Intelligent RAG Sandbox",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Rich Aesthetics)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-sql {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #90CAF9;
    }
    .badge-rag {
        background-color: #F3E5F5;
        color: #7B1FA2;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #CE93D8;
    }
    .badge-model {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #A5D6A7;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Persistent Database & Embedding Model Caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def init_db_connection():
    conn = get_db_connection()
    ensure_feedback_table(conn)
    return conn

model = load_embedding_model()
conn = init_db_connection()

# Sidebar Navigation
st.sidebar.title("InvoiceInsight")
st.sidebar.caption("Hybrid Text-to-SQL + RAG Architecture")

page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat Assistant", "📊 Monitoring Dashboard", "🗄️ Database Inspector"],
    index=0
)

st.sidebar.divider()

# Sidebar System Configuration
st.sidebar.subheader("⚙️ System Settings")

# Model Selection Options: Primary default, Extra Llama, and Budget model
available_models = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

model_choice = st.sidebar.selectbox(
    "LLM Model",
    available_models,
    index=0,
    help="Select the AI model used for Text-to-SQL generation and RAG answer synthesis."
)



st.sidebar.divider()
st.sidebar.caption("LLM Zoomcamp Capstone Sandbox v2.0")

# ==========================================
# PAGE 1: CHAT ASSISTANT
# ==========================================
if page == "💬 Chat Assistant":
    st.markdown('<div class="main-header">InvoiceInsight Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask structured financial questions or perform fuzzy document discovery across 200 ingested invoices.</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am InvoiceInsight. Ask me anything about your invoices (e.g. *'What is the billing date for INV-2026-0145?'* or *'Which vendor supplied office chairs?'*)."
            }
        ]

    # Render Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display method and model badge if available
            if "method" in msg:
                if "sql" in msg["method"].lower():
                    st.markdown(f'<span class="badge-sql">⚡ Method: {msg["method"].upper()}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-rag">🔍 Method: {msg["method"].upper()}</span>', unsafe_allow_html=True)
                
                if msg.get("llm_model"):
                    st.markdown(f'<span class="badge-model">🤖 {msg["llm_model"]}</span>', unsafe_allow_html=True)
            
            # Display expandable execution details
            if "details" in msg and msg["details"]:
                with st.expander("🔍 View Execution Details & Context"):
                    if "sql" in msg.get("method", "").lower():
                        st.code(msg["details"].get("sql", "N/A"), language="sql")
                        if msg["details"].get("data"):
                            st.json(msg["details"]["data"][:5])
                    else:
                        chunks = msg["details"].get("retrieved_chunks") or []
                        for idx_c, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Document {idx_c}** — `Invoice: {chunk.get('invoice_id')}` (Similarity: {chunk.get('similarity', 0):.4f})")
                            st.caption(chunk.get("content_text")[:200] + "...")

    # Chat Input Form
    if user_query := st.chat_input("Type your invoice query here..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Processing query through hybrid router..."):
                start_time = time.time()
                conn = get_healthy_db_connection(conn)
                
                try:
                    res = answer_query(
                        user_query, 
                        model, 
                        conn, 
                        llm_model=model_choice
                    )
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    
                    answer_text = res.get("answer", "No response generated.")
                    method = res.get("method", "rag")
                    
                    st.markdown(answer_text)
                    
                    # Display Method & Model Badges
                    if "sql" in method.lower():
                        st.markdown(f'<span class="badge-sql">⚡ Method: {method.upper()}</span>', unsafe_allow_html=True)
                        retrieved_ids = [res.get("sql")] if res.get("sql") else []
                    else:
                        st.markdown(f'<span class="badge-rag">🔍 Method: {method.upper()}</span>', unsafe_allow_html=True)
                        chunks = res.get("retrieved_chunks") or []
                        retrieved_ids = [c.get("invoice_id") for c in chunks if c.get("invoice_id")]
                    
                    st.markdown(f'<span class="badge-model">🤖 {model_choice}</span>', unsafe_allow_html=True)
                    st.caption(f"⏱️ Response Latency: `{elapsed_ms} ms`")

                    # Automatically Log Query Telemetry Record into PostgreSQL
                    row_id = log_query_telemetry(conn, user_query, answer_text, method, retrieved_ids, elapsed_ms, user_rating=0)

                    # Expandable Execution Details
                    with st.expander("🔍 View Execution Details & Context"):
                        if "sql" in method.lower():
                            st.subheader("Generated PostgreSQL Query")
                            st.code(res.get("sql", "N/A"), language="sql")
                            if res.get("data"):
                                st.subheader("Query Result Payload")
                                st.json(res["data"][:5])
                        else:
                            st.subheader("Top Retrieved Context Chunks")
                            chunks = res.get("retrieved_chunks") or []
                            for idx_c, chunk in enumerate(chunks, 1):
                                st.markdown(f"**Document {idx_c}** — `Invoice: {chunk.get('invoice_id')}` (Similarity: {chunk.get('similarity', 0):.4f})")
                                st.text(chunk.get("content_text"))

                    # Store message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "method": method,
                        "llm_model": model_choice,
                        "details": res,
                        "elapsed_ms": elapsed_ms,
                        "query": user_query,
                        "retrieved_ids": retrieved_ids,
                        "db_id": row_id
                    })

                    # Interactive Feedback Logger Buttons
                    col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 8])
                    with col_fb1:
                        if st.button("👍", key=f"like_{len(st.session_state.messages)}"):
                            update_feedback_rating(conn, row_id, 1)
                            st.toast("Thank you for your feedback! (+1 Recorded)")
                    with col_fb2:
                        if st.button("👎", key=f"dislike_{len(st.session_state.messages)}"):
                            update_feedback_rating(conn, row_id, -1)
                            st.toast("Thank you for your feedback! (-1 Recorded)")

                except Exception as e:
                    st.error(f"Error executing pipeline: {str(e)}")

# ==========================================
# PAGE 2: MONITORING DASHBOARD
# ==========================================
elif page == "📊 Monitoring Dashboard":
    st.markdown('<div class="main-header">System Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time performance analytics, query method distribution, latency metrics, and conversation logs.</div>', unsafe_allow_html=True)

    conn = get_healthy_db_connection(conn)
    df_feedback = fetch_feedback_data(conn)

    if df_feedback.empty:
        st.info("ℹ️ No query telemetry recorded yet. Interact with the **💬 Chat Assistant** to populate the monitoring dashboard!")
    else:
        # Top KPI Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_queries = len(df_feedback)
        likes = (df_feedback["user_rating"] == 1).sum()
        dislikes = (df_feedback["user_rating"] == -1).sum()
        rated_queries = likes + dislikes
        satisfaction_pct = (likes / rated_queries * 100) if rated_queries > 0 else 100.0
        avg_latency = df_feedback["response_time_ms"].mean() if total_queries > 0 else 0
        sql_pct = (df_feedback["method"].str.contains("sql", case=False).sum() / total_queries * 100) if total_queries > 0 else 0

        with col1:
            st.metric("Total Queries Answered", f"{total_queries}")
        with col2:
            st.metric("User Satisfaction Rate", f"{satisfaction_pct:.1f}%", f"{likes} Likes / {dislikes} Dislikes")
        with col3:
            st.metric("Avg Response Time", f"{avg_latency:.0f} ms")
        with col4:
            st.metric("SQL Router Usage", f"{sql_pct:.1f}%", f"{100-sql_pct:.1f}% RAG")

        st.divider()

        # Row 1: 2 Pie Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("1. User Feedback Rating Distribution")
            rating_map = {1: "👍 Positive (+1)", -1: "👎 Negative (-1)", 0: "💬 Unrated (0)"}
            fb_counts = df_feedback["user_rating"].map(rating_map).value_counts().reset_index()
            fb_counts.columns = ["Rating", "Count"]
            fig_fb = px.pie(fb_counts, names="Rating", values="Count", color="Rating",
                            color_discrete_map={
                                "👍 Positive (+1)": "#2E7D32",
                                "👎 Negative (-1)": "#C62828",
                                "💬 Unrated (0)": "#9E9E9E"
                            }, hole=0.4)
            st.plotly_chart(fig_fb, use_container_width=True)

        with col_c2:
            st.subheader("2. Query Method Distribution (SQL vs RAG)")
            method_counts = df_feedback["method"].str.upper().value_counts().reset_index()
            method_counts.columns = ["Method", "Count"]
            fig_method = px.pie(method_counts, names="Method", values="Count", color="Method",
                                hole=0.4)
            st.plotly_chart(fig_method, use_container_width=True)

        # Row 2: Response Time & Top Invoices
        col_c3, col_c4 = st.columns(2)

        with col_c3:
            st.subheader("3. Response Latency Trend (ms)")
            fig_latency = px.line(df_feedback, x="created_at", y="response_time_ms", color="method",
                                  markers=True, title="Query Response Latency Over Time")
            st.plotly_chart(fig_latency, use_container_width=True)

        with col_c4:
            st.subheader("4. Top Retrieved Invoice IDs")
            all_ids = []
            for item in df_feedback["retrieved_ids"].dropna():
                if isinstance(item, list):
                    all_ids.extend(item)
                elif isinstance(item, str):
                    all_ids.append(item)
            if all_ids:
                df_ids = pd.Series(all_ids).value_counts().head(8).reset_index()
                df_ids.columns = ["Invoice ID", "Frequency"]
                fig_ids = px.bar(df_ids, x="Frequency", y="Invoice ID", orientation="h", color="Frequency")
                st.plotly_chart(fig_ids, use_container_width=True)
            else:
                st.info("No retrieved IDs recorded yet.")

        # Row 3: Daily Query Volume
        st.subheader("5. Daily Query Volume")
        df_feedback["date"] = pd.to_datetime(df_feedback["created_at"]).dt.date
        daily_counts = df_feedback.groupby("date").size().reset_index(name="Query Count")
        fig_daily = px.bar(daily_counts, x="date", y="Query Count", title="Total Queries Processed Per Day")
        st.plotly_chart(fig_daily, use_container_width=True)

        # Raw Conversation Audit Log Table
        st.subheader("📜 Complete Conversation & Telemetry Audit Log")
        st.dataframe(
            df_feedback[["id", "created_at", "method", "user_rating", "response_time_ms", "question", "answer"]],
            use_container_width=True
        )

# ==========================================
# PAGE 3: DATABASE INSPECTOR
# ==========================================
elif page == "🗄️ Database Inspector":
    st.markdown('<div class="main-header">PostgreSQL Vector Database Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect stored invoice chunks, markdown text, and raw JSON payloads inside PostgreSQL (`invoice_chunks`).</div>', unsafe_allow_html=True)

    conn = get_healthy_db_connection(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM invoice_chunks;")
    row = cursor.fetchone()
    total_db_chunks = list(row.values())[0] if row else 0
    cursor.close()

    st.success(f"✅ Total Ingested Invoice Chunks in Database: **{total_db_chunks} Records**")

    search_term = st.text_input("Filter records by Invoice ID or Vendor:", "")

    cursor = conn.cursor()
    if search_term:
        cursor.execute("""
            SELECT id, invoice_id, content_json->>'vendor' as vendor, content_json->>'date' as date,
                   content_json->>'grand_total' as total, content_json->>'currency' as currency
            FROM invoice_chunks
            WHERE invoice_id ILIKE %s OR content_json->>'vendor' ILIKE %s
            ORDER BY id ASC
            LIMIT 50;
        """, (f"%{search_term}%", f"%{search_term}%"))
    else:
        cursor.execute("""
            SELECT id, invoice_id, content_json->>'vendor' as vendor, content_json->>'date' as date,
                   content_json->>'grand_total' as total, content_json->>'currency' as currency
            FROM invoice_chunks
            ORDER BY id ASC
            LIMIT 50;
        """)
    
    db_rows = cursor.fetchall()
    cursor.close()

    if db_rows:
        df_db = pd.DataFrame(db_rows)
        st.dataframe(df_db, use_container_width=True)
    else:
        st.warning("No records found matching search filter.")
