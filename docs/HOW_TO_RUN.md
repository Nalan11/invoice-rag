# Developer Reference & Run Guide: InvoiceInsight RAG System

This guide serves as a complete reference for running, testing, and debugging the **InvoiceInsight** Hybrid Text-to-SQL + Vector RAG system locally.

---

## 1. System Architecture Overview

The system consists of three main components:
1. **PostgreSQL 16 + pgvector (`invoice_insight_db`)**: Runs in Docker on port `5432`. Stores structured invoice JSON, markdown serialisations, 384-dimensional vector embeddings, and query telemetry.
2. **Hybrid RAG Pipeline (`src/`)**:
   - **Router (`router.py`)**: Intent classification (SQL aggregation vs. Vector semantic search vs. Hybrid).
   - **Text-to-SQL Engine (`sql_engine.py`)**: Safe parameterized JSONB query generator.
   - **Vector Search (`search.py` & `rag.py`)**: Cosine similarity retrieval over `all-MiniLM-L6-v2` embeddings.
   - **LLM Client (`config.py`)**: OpenAI-compatible client with built-in 429 rate limit backoff.
3. **Streamlit UI (`streamlit_app.py`)**: Web application featuring chat interface, SQL debug drawers, and telemetry monitoring dashboard.

---

## 2. Prerequisites

- **Python**: Version `3.11.x` recommended.
- **Docker Desktop**: Must be installed and running.
- **Git**: For version control.
- **Groq API Key**: (or any OpenAI-compatible endpoint key).

---

## 3. Step-by-Step Setup & Run Commands

### Step A: Configure Environment Variables
Ensure you have a `.env` file in the project root:

```ini
# LLM Provider (Groq or OpenAI-compatible)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_your_actual_api_key_here
LLM_MODEL=openai/gpt-oss-20b

# PostgreSQL Database Connection
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/invoice_insight
```

> **Note:** Never commit `.env` to Git. A clean template is available in `.env.example`.

---

### Step B: Start the Database Container

Start the PostgreSQL + `pgvector` container in the background:
```powershell
docker compose up -d
```

Verify the container is healthy and running:
```powershell
docker ps
```
*(You should see `invoice_insight_db` mapped to port `0.0.0.0:5432->5432/tcp`)*

To view database logs in real time:
```powershell
docker logs -f invoice_insight_db
```

---

### Step C: Virtual Environment & Dependencies

If you are using the existing `venv`:
```powershell
# In PowerShell:
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run this once):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

If you ever need to recreate the environment from scratch:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Step D: Database Ingestion (Load Invoices & Embeddings)

If starting with a fresh database or after resetting the Docker volume, populate the 200 invoice records and compute vector embeddings:

```powershell
python 02_ingestion.py
```
*(This serializes `data/invoices.json`, encodes using `all-MiniLM-L6-v2`, and upserts all 200 records into the `invoice_chunks` table).*

To regenerate fresh synthetic invoice data and ground truth:
```powershell
python data/generate_invoices.py
python data/generate_ground_truth.py
python 02_ingestion.py
```

---

### Step E: Launch the Streamlit Web Application

Run the interactive web UI:
```powershell
streamlit run streamlit_app.py
```

Once running:
- **Local URL**: `http://localhost:8501`
- **Features**:
  - **Assistant Tab**: Ask complex financial, vendor, or item queries. View live latency and SQL queries in expandable drawers. Give user ratings (thumbs up/down).
  - **Monitoring Dashboard Tab**: Live KPIs (total queries, avg latency, satisfaction rate) and feedback telemetry.

---

## 4. Evaluation & Testing Commands

### 1. Critical Adversarial Stress Test (30 Hard Questions)
Tests edge cases (ambiguous totals, missing invoices, injection attempts, multi-criteria filters):
```powershell
python run_critical_test.py
```
*(Outputs summary table and saves detailed results to `critical_test_results.json`).*

### 2. Batch Evaluation Suite
Runs a quick batch of common invoice queries:
```powershell
python run_batch_test.py
```

### 3. Retrieval Evaluation (Hit Rate & MRR)
Evaluates top-k semantic search performance against ground-truth pairs:
```powershell
python 03_retrieval_evaluation.py
```

### 4. End-to-End Pipeline Evaluation (LLM-as-a-Judge)
Scores answer correctness against reference ground truth:
```powershell
python 04_pipeline_evaluation.py
```

---

## 5. Telemetry & History Inspection

To quickly inspect what questions were asked, user ratings, and latency without opening the web browser:

```powershell
# View the last 10 queries asked:
python inspect_last_10.py

# View full logged query history:
python inspect_history.py
```

---

## 6. Maintenance & Troubleshooting Cheatsheet

| Issue / Goal | Command |
| :--- | :--- |
| **Port 5432 conflict** | Another Postgres instance is running locally. Stop local Postgres service (`Stop-Service postgresql*`) or change host port in `docker-compose.yml`. |
| **Stop Database** | `docker compose stop` |
| **Complete Database Reset** (Wipes all data & starts clean) | `docker compose down -v`<br>`docker compose up -d`<br>`python 02_ingestion.py` |
| **Database Container Shell** | `docker exec -it invoice_insight_db psql -U postgres -d invoice_insight` |
| **Check Table Counts** | `docker exec -it invoice_insight_db psql -U postgres -d invoice_insight -c "SELECT count(*) FROM invoice_chunks; SELECT count(*) FROM feedback;"` |
| **LLM 429 Rate Limit** | The system automatically waits (5s -> 10s -> 20s). If daily quota is hit, switch model in Streamlit sidebar or update `LLM_MODEL` in `.env`. |
| **Commit latest work to GitHub** | `git add .`<br>`git commit -m "your message"`<br>`git push origin main` |
