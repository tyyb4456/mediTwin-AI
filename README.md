# 🧠 MediTwin AI — Backend Microservices

> A multi-agent AI clinical decision support platform built with FastAPI, LangGraph, LangChain, Google Gemini, ChromaDB, Redis, and PostgreSQL.

---

## 📐 Architecture Overview

MediTwin AI is a **microservices monorepo** — all agents are built from a single shared codebase and each runs as an independent FastAPI service.

```
meditwin-ai/
├── agents/
│   ├── patient_context/     # Port 8001 — FHIR patient data enrichment
│   ├── diagnosis/           # Port 8002 — RAG-powered differential diagnosis
│   ├── lab_analysis/        # Port 8003 — Lab result interpretation
│   ├── drug_safety/         # Port 8004 — Drug interaction checker (MCP)
│   ├── imaging_triage/      # Port 8005 — Medical image triage
│   ├── digital_twin/        # Port 8006 — XGBoost patient simulation
│   ├── consensus/           # Port 8007 — Multi-agent consensus builder
│   ├── explanation/         # Port 8009 — Plain-language explanation generator
│   └── tool_agent/          # Port 8010 — LangGraph tool-calling orchestrator
├── orchestrator/            # Port 8000 — Main streaming workflow entry point
├── shared/                  # Shared utilities (auth, models, etc.)
├── knowledge_base/          # Medical knowledge ingest for ChromaDB (RAG)
├── docker-compose.yml
├── requirements.txt
└── .env
```

### Service Port Map

| Service | Port | Description |
|---|---|---|
| Orchestrator | `8000` | Main entry point — drives the full workflow |
| Patient Context | `8001` | FHIR-based patient enrichment |
| Diagnosis | `8002` | RAG differential diagnosis |
| Lab Analysis | `8003` | Lab result interpretation |
| Drug Safety | `8004` | MCP-based drug interaction checks |
| Imaging Triage | `8005` | Medical image analysis |
| Digital Twin | `8006` | XGBoost patient simulation |
| Consensus | `8007` | Multi-agent vote & consensus |
| Explanation | `8009` | Plain-English explanations |
| Tool Agent | `8010` | Conversational LangGraph agent |
| ChromaDB | `8008` | Vector store (internal) |
| Redis | `6379` | Cache layer (internal) |
| PostgreSQL | `5432` | LangGraph checkpointer (internal) |

---

## 🚀 Quick Start — Docker (Recommended)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.20 (bundled with Docker Desktop)
- A **Google Gemini API key** — [Get one free here](https://aistudio.google.com/app/apikey)

### Step 1 — Clone & navigate

```bash
git clone <your-repo-url>
cd meditwin-ai
```

### Step 2 — Configure environment

Create your `.env` file from the example below. **Never commit this file.**

```bash
# .env
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Option A: Use the bundled local PostgreSQL (default for Docker)
# Leave this unset — docker-compose handles it automatically.

# Option B: Use an external PostgreSQL (e.g. Railway, Supabase)
# POSTGRES_CHECKPOINT_URI=postgresql://user:password@host:port/dbname
```

> ⚠️ **Important:** If you skip the `.env` file or leave `GOOGLE_API_KEY` empty, all LLM-dependent agents will fail to start.

### Step 3 — One-time setup (first run only)

These commands seed the knowledge base (ChromaDB) and train the Digital Twin ML models. **Run these before the main stack.**

```bash
# 1. Ingest medical knowledge into ChromaDB (for RAG)
docker-compose run --rm diagnosis-ingest

# 2. Train the Digital Twin XGBoost models
docker-compose run --rm digital-twin-train
```

> ⏱ **Expected time:** ~2–5 min depending on your internet speed and hardware.

### Step 4 — Start the full stack

```bash
docker-compose up --build
```

On subsequent runs (no code changes):

```bash
docker-compose up
```

### Step 5 — Verify services are healthy

Once all containers are running, open your browser or use `curl`:

```bash
# Orchestrator health check
curl http://localhost:8000/health

# Diagnosis agent
curl http://localhost:8002/health

# Tool agent (conversational)
curl http://localhost:8010/health
```

All endpoints return `{ "status": "ok" }` when healthy.

---

## 🔧 Local Development (Without Docker)

Use this approach when you want to run individual agents without the full stack — faster iteration for single-agent development.

### Prerequisites

- Python 3.11+
- Redis (local install or use Docker for Redis only)
- PostgreSQL (local or remote)
- A Google Gemini API key

### Step 1 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Start infrastructure services (Redis + ChromaDB) via Docker

You don't need the full compose stack — just the infrastructure:

```bash
docker-compose up redis chromadb postgres-checkpoint -d
```

### Step 4 — Configure environment variables

Copy the block below into a `.env` file at the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
CHROMADB_HOST=localhost
CHROMADB_PORT=8008
POSTGRES_CHECKPOINT_URI=postgresql://postgres:postgres@localhost:5432/meditwin_checkpoints
FHIR_BASE_URL=https://hapi.fhir.org/baseR4
INTERNAL_TOKEN=meditwin-internal
```

### Step 5 — Ingest the knowledge base (one-time)

```bash
python knowledge_base/ingest.py
```

### Step 6 — Train Digital Twin models (one-time)

```bash
python agents/digital_twin/train_models.py
```

### Step 7 — Run individual agents

Each agent is a standalone FastAPI/uvicorn app. Open separate terminal windows for each:

```bash
# Patient Context Agent
cd agents/patient_context
uvicorn main:app --port 8001 --reload

# Diagnosis Agent
cd agents/diagnosis
uvicorn main:app --port 8002 --reload

# Lab Analysis Agent
cd agents/lab_analysis
uvicorn main:app --port 8003 --reload

# Drug Safety Agent
cd agents/drug_safety
uvicorn main:app --port 8004 --reload

# Imaging Triage Agent
cd agents/imaging_triage
uvicorn main:app --port 8005 --reload

# Digital Twin Agent
cd agents/digital_twin
uvicorn main:app --port 8006 --reload

# Consensus Agent
cd agents/consensus
uvicorn main:app --port 8007 --reload

# Explanation Agent
cd agents/explanation
uvicorn main:app --port 8009 --reload

# Orchestrator (depends on all agents above)
cd orchestrator
uvicorn main:app --port 8000 --reload

# Tool Agent (conversational interface)
cd agents/tool_agent
uvicorn main:app --port 8010 --reload
```

> 💡 **Tip:** Use [tmux](https://github.com/tmux/tmux) or [Windows Terminal](https://aka.ms/terminal) tabs to manage multiple agent processes.

---

## 🐳 Docker — Useful Commands

### Rebuild a single service

```bash
docker-compose build --no-cache diagnosis
docker-compose up diagnosis
```

### Rebuild all without cache

```bash
docker-compose build --no-cache
docker-compose up
```

### View logs for a specific agent

```bash
docker-compose logs -f diagnosis
docker-compose logs -f orchestrator
docker-compose logs -f tool-agent
```

### Stop everything

```bash
docker-compose down
```

### Stop and wipe all volumes (full reset)

```bash
docker-compose down -v
```

> ⚠️ This deletes ChromaDB and PostgreSQL data. You will need to re-run the one-time setup steps.

### Shell into a running container

```bash
docker exec -it meditwin-diagnosis bash
docker exec -it meditwin-orchestrator bash
```

---

## 🌐 API Reference (Key Endpoints)

### Orchestrator — `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze/stream` | Stream full multi-agent workflow (SSE) |
| `GET` | `/health` | Health check |

### Diagnosis Agent — `http://localhost:8002`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/diagnose/stream` | Stream RAG diagnosis (SSE) |
| `GET` | `/history/{thread_id}` | Get conversation history |
| `DELETE` | `/history/{thread_id}` | Clear conversation history |

### Tool Agent — `http://localhost:8010`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query/stream` | Conversational agent with tool calls (SSE) |

### Lab Analysis — `http://localhost:8003`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze/stream` | Stream lab analysis (SSE) |

### Drug Safety — `http://localhost:8004`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/check/stream` | Drug interaction stream (SSE) |

### Digital Twin — `http://localhost:8006`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/simulate/stream` | Patient simulation stream (SSE) |

---

## 🔑 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | — | Google Gemini API key |
| `POSTGRES_CHECKPOINT_URI` | ✅ Yes | local postgres | LangGraph checkpointer DB URI |
| `REDIS_HOST` | ✅ Local only | `redis` | Redis hostname |
| `REDIS_PORT` | No | `6379` | Redis port |
| `CHROMADB_HOST` | ✅ Local only | `chromadb` | ChromaDB hostname |
| `CHROMADB_PORT` | No | `8000` | ChromaDB internal port |
| `FHIR_BASE_URL` | No | HAPI FHIR | FHIR server base URL |
| `INTERNAL_TOKEN` | No | `meditwin-internal` | Inter-service auth token |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11 |
| Web Framework | FastAPI + Uvicorn |
| LLM | Google Gemini (via LangChain) |
| Agent Framework | LangGraph |
| Vector Store | ChromaDB |
| Cache | Redis |
| Checkpointer | PostgreSQL (LangGraph) |
| ML Models | XGBoost, scikit-learn |
| Medical Data | FHIR (HAPI R4) |
| Containerization | Docker + Docker Compose |

---

## ❓ Troubleshooting

**`diagnosis` service exits immediately**
→ Check that `diagnosis-ingest` completed successfully before starting the main stack.

**`digital-twin` or `explanation` service fails**
→ Ensure `digital-twin-train` ran to completion. Models must exist before these services start.

**`GOOGLE_API_KEY` errors**
→ Make sure `.env` is in the project root and the key is valid.

**Port already in use**
→ Another process is using one of the ports. Kill it or change the port mapping in `docker-compose.yml`.

**ChromaDB collection not found**
→ Re-run: `docker-compose run --rm diagnosis-ingest`

**PostgreSQL connection refused**
→ Ensure `postgres-checkpoint` container is running: `docker-compose up postgres-checkpoint -d`
