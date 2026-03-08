# 🔍 RCA Agent — LLM-Powered Log Analysis

> Ask questions about your system logs in plain English.
> Get structured root cause analysis in seconds.

---

## 🔥 The Problem

When production systems fail at 2am, engineers spend hours manually
searching through thousands of log lines across multiple services
to find the actual cause. A junior engineer might not find it at all.

**This project solves that.**

Instead of grepping through logs, an engineer types:

> *"Why did the payment service crash at 2am?"*

And gets back:
```
1. 🔍 ROOT CAUSE
   v2.4.0 deployment introduced an unbounded memory cache.
   Traffic spike at 1:44am filled it rapidly.

2. 🔗 CHAIN OF EVENTS
   23:10 → v2.4.0 deployed (ResponseCache.java, no max-size)
   01:44 → Traffic spike 1240 req/s (3.1x baseline)
   01:58 → DB connection pool exhausted (50/50)
   02:00 → OOMKilled (exit code 137)

3. 🚨 IMMEDIATE FIX
   Roll back to v2.3.1 or increase memory limit to 2Gi

4. 🛡️ LONG TERM FIX
   Set cache max-size. Add memory alerts. Load test before deploy.
```

Hours of investigation → seconds. Expert knowledge → accessible to everyone.

---

## 🏗️ Architecture
```
LOG FILES (logs/)
      │
      ▼
CHUNKER (400 char chunks, 80 overlap)
      │
      ▼
EMBEDDING MODEL (all-MiniLM-L6-v2, local, free)
      │
      ▼
CHROMADB (persisted to data/chroma_db/)
      │
      │ ◄── User asks a question
      │
      ▼
RETRIEVER (top-5 similar chunks)
      │
      ├──► SIMPLE RAG MODE
      │    └── chunks + question → LLM → structured answer
      │
      └──► REACT AGENT MODE
           └── LLM decides tools → search_logs / check_metrics
               / get_deployments → loops until final answer
```

---

## 🧠 Two Modes

### Simple RAG
One similarity search → one LLM call → answer.
Fast. Good for focused questions.
```
Question → embed → ChromaDB search → prompt + chunks → LLM → answer
```

### ReAct Agent
Multi-step reasoning. The LLM decides what to investigate next.
```
Question → THOUGHT → ACTION (tool) → OBSERVATION
        → THOUGHT → ACTION (tool) → OBSERVATION
        → ... repeat until ...
        → FINAL ANSWER
```

The agent has 3 tools:
- `search_logs` — semantic search over ChromaDB
- `check_metric_spike` — memory/CPU/traffic metrics
- `get_deployment_history` — recent deployments and code changes

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| Framework | LangChain | Connects all AI pieces together |
| LLM | Llama 3.3 70B via Groq | Free, fast (~300 tok/s), open source |
| Embeddings | all-MiniLM-L6-v2 | Free, runs locally, 384 dimensions |
| Vector Store | ChromaDB | Free, runs locally, persists to disk |
| UI | Streamlit | Web app in pure Python |
| Agent Pattern | ReAct | Reasoning + Acting loop |

**Everything is free. No credit card needed.**

---

## 🚀 Setup

### Prerequisites
- Python 3.11
- Free Groq API key (console.groq.com)

### Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rca-agent.git
cd rca-agent

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=gsk_xxxx
```

### Build The Index (run once)
```bash
python src/1_ingest_logs.py
```

This reads the log files, converts them to embeddings, and stores
them in ChromaDB. Only needs to run once (or when you add new logs).

### Launch The App
```bash
streamlit run src/4_streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 💬 Example Questions

Try these in the app:

- *"Why did the payment service crash at 2am?"*
- *"What caused the database connection errors?"*
- *"Was there a deployment before the incident?"*
- *"What is the full timeline of the failure?"*
- *"How was the service recovered?"*

---

## 📁 Project Structure
```
rca-agent/
│
├── .env                      ← Groq API key (never commit this)
├── .gitignore
├── requirements.txt
├── README.md
│
├── logs/
│   ├── k8s_payment_service.log   ← Kubernetes pod logs
│   └── postgres_database.log     ← Database logs
│
├── data/
│   └── chroma_db/            ← Vector store (auto-created)
│
└── src/
    ├── 1_ingest_logs.py      ← Load → chunk → embed → store
    ├── 2_rag_pipeline.py     ← Simple RAG chain
    ├── 3_agent.py            ← ReAct agent with tools
    └── 4_streamlit_app.py    ← Streamlit UI
```

---

## 🧩 Key Concepts

**Embeddings** — Text converted to numbers where similar meanings
produce similar numbers. Allows semantic search rather than
keyword matching.

**RAG (Retrieval Augmented Generation)** — Retrieve relevant
context first, then give it to the LLM with the question.
Prevents hallucination by grounding answers in real log data.
Like an open book exam for the AI.

**ReAct Agent** — LLM that loops through Thought → Action →
Observation using tools. Reasons like a senior engineer
investigating an incident — forms hypotheses and gathers
evidence across multiple sources.

**Vector Store** — Specialized database that stores embeddings
and finds similar ones fast using cosine similarity.
Unlike regular databases that match exact values.

---

## 🏭 Production Considerations

| Component | Demo | Production |
|---|---|---|
| Log Ingestion | Manual file load | Kafka/Kinesis streaming |
| Vector Store | ChromaDB local | Pinecone / pgvector |
| Retrieval | Dense only | Hybrid BM25 + dense |
| Metrics Tool | Simulated data | Live Prometheus API |
| LLM | Groq free tier | Azure OpenAI / AWS Bedrock |
| Observability | Console logs | LangSmith tracing |
