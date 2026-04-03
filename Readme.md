# MediTwin AI Backend - Build Progress

**Project:** Multi-Agent Clinical Decision Support System  
**Strategy Document:** `Meditwin backend strategy.md`  
**Build Approach:** One agent at a time, test, then move forward

---

## Build Progress

### ✅ Phase 1: Infrastructure Setup (COMPLETE)
- [x] Docker Compose configuration
- [x] Network setup for all services
- [x] Redis for caching
- [x] ChromaDB for vector storage
- [x] Shared modules (models, Redis client, SHARP context)
- [x] Requirements.txt with all dependencies

### ✅ Phase 2: Agent 1 - Patient Context Agent (COMPLETE - NOT TESTED YET)
- [x] FastAPI app with async httpx
- [x] SHARP context header parsing
- [x] Parallel FHIR resource fetching
- [x] Patient data normalization
- [x] Redis caching (10-minute TTL)
- [x] Dockerfile
- [x] Test script

**Location:** `agents/patient_context/`

**What it does:**
- Fetches FHIR R4 resources from HAPI FHIR sandbox
- Normalizes to PatientState Pydantic model
- Caches in Redis
- Handles SHARP context from Prompt Opinion platform

**Next:** Test the agent before building next one

---

## Testing Patient Context Agent

### Option 1: Standalone Test (Requires Redis)

```bash
# Terminal 1: Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 2: Run the agent
cd meditwin/agents/patient_context
python main.py

# Terminal 3: Run test script
python test.py
```

### Option 2: Full Docker Compose (Later)

```bash
docker-compose up patient-context redis
```

---

## Next Steps (Build Order)

According to strategy document, the correct build sequence is:

1. ✅ Infrastructure → Patient Context 
2. **NEXT:** Diagnosis Agent + Lab Analysis Agent (can build in parallel)
3. Drug Safety MCP
4. Imaging Triage Agent  
5. Digital Twin Agent
6. Consensus + Escalation (LangGraph node)
7. Explanation Agent
8. Orchestrator (LAST - ties everything together)

---

## Key Design Decisions Made

### 1. Async-First Architecture
- Using `httpx.AsyncClient` with proper lifespan management
- All FHIR fetches are parallel using `asyncio.gather()`
- Follows latest FastAPI best practices (2024-2026)

### 2. Shared Code Strategy
- `/shared/models.py` - PatientState is the canonical data contract
- `/shared/redis_client.py` - Singleton Redis client
- `/shared/sharp_context.py` - SHARP header parsing

### 3. Cache Strategy
- Patient Context: 10-minute TTL
- Drug Safety: 1-hour TTL (not built yet)
- Cache keys: `patient_state:{patient_id}`

### 4. Error Handling
- Graceful handling of missing FHIR resources
- Returns empty arrays for missing data
- HTTP exceptions with clear error messages

---

## File Structure

```
meditwin/
├── docker-compose.yml          # All services
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
│
├── shared/                    # Shared across all agents
│   ├── models.py             # Pydantic models (PatientState, etc.)
│   ├── redis_client.py       # Async Redis wrapper
│   └── sharp_context.py      # SHARP header parsing
│
├── agents/
│   └── patient_context/      # ✅ Agent 1 (BUILT)
│       ├── main.py           # FastAPI app
│       ├── Dockerfile        # Container config
│       └── test.py           # Test script
│
├── orchestrator/             # To be built last
└── knowledge_base/           # ChromaDB data
    └── sources/             # Medical documents
```

---

## Critical Reminders

1. **PatientState is sacred** - Changing its schema breaks all downstream agents
2. **Test before proceeding** - Each agent must work before building the next
3. **SHARP context optional** - Agents work with both SHARP headers and direct input
4. **Parallel where possible** - Diagnosis + Lab have no dependency, build together

---

## API Keys Required

```bash
# Copy .env.example to .env and fill in:
OPENAI_API_KEY=sk-...  # Required for Diagnosis, Lab, Drug Safety, Explanation
```

---

## What Agent 1 Validates

Before moving to Agent 2, we must confirm:

- [x] Code compiles and runs
- [ ] Health endpoint returns 200
- [ ] Can fetch a real patient from HAPI FHIR
- [ ] PatientState has all required fields populated
- [ ] Second identical call hits Redis cache (< 10ms response)
- [ ] Missing FHIR resources don't crash the agent

**Status:** Code complete, awaiting first test run

---

## Next Agent: Diagnosis Agent

**Dependencies:** ChromaDB, OpenAI API  
**Key Components:**
- RAG pipeline over medical knowledge base
- LangChain for structured output
- Differential diagnosis ranking
- FHIR Condition resource builder

**Before building:** Must seed ChromaDB with medical documents