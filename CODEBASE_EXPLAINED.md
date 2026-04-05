# MediTwin AI — Codebase Explained

> A plain-English walkthrough of the entire codebase for developers.  
> No medical degree required.

---

## What Is This Project?

MediTwin AI is a **multi-agent microservice system** that acts as a clinical decision support tool. Think of it like a pipeline of 9 specialized AI "workers" — each one does one job really well, and they all pass data to each other to produce a final medical analysis for a patient.

**The big idea:** You give it a patient ID → it fetches the patient's health records → runs them through diagnosis, lab analysis, imaging, drug safety checks, risk simulation, conflict resolution, and finally produces a human-readable report + a standards-compliant data bundle.

**Tech stack:**
- **Python + FastAPI** — each agent is its own HTTP microservice
- **LangChain + Google Gemini** — LLM-powered reasoning (diagnosis, interpretation, narratives)
- **ChromaDB** — vector database for RAG (retrieval-augmented generation) over medical guidelines
- **Redis** — caching layer (patient data + drug safety results)
- **XGBoost** — ML models for risk prediction (digital twin)
- **TensorFlow/Keras** — CNN model for chest X-ray analysis
- **LangGraph** — state machine that orchestrates the entire pipeline
- **Docker Compose** — everything runs in containers on a shared network
- **FHIR R4** — healthcare data standard used for all inputs/outputs

---

## Architecture at a Glance

```
                        ┌──────────────┐
         User Request → │ Orchestrator │ ← The only public-facing service (port 8000)
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌────────────────┐ ┌───────────┐  ┌──────────────┐
     │ Patient Context│ │ Diagnosis │  │ Lab Analysis  │
     │   (port 8001)  │ │ (port 8002│  │  (port 8003)  │
     └────────┬───────┘ └─────┬─────┘  └──────┬───────┘
              │               │                │
              │        ┌──────┴────────────────┘
              ▼        ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Drug Safety   │ │   Imaging    │ │ Digital Twin  │
     │  (port 8004)   │ │  (port 8005) │ │  (port 8006)  │
     └────────┬───────┘ └──────┬───────┘ └──────┬───────┘
              │                │                 │
              └────────┬───────┴─────────────────┘
                       ▼
              ┌────────────────┐
              │   Consensus    │  ← Detects disagreements between agents
              │  + Escalation  │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │  Explanation   │  ← Turns everything into readable reports
              │  (port 8007)   │
              └────────────────┘
```

**Data flows top-down.** The Orchestrator calls agents in order, passing each agent's output into the next. Some agents run in parallel (Diagnosis + Lab Analysis). The Consensus agent checks if everyone agrees. The Explanation agent packages the final output.

---

## Shared Infrastructure

Before diving into agents, here's what ties them all together:

### `shared/models.py` — The Data Contracts

Defines Pydantic models that every agent uses. The most important one is **`PatientState`** — a structured object containing:
- **Demographics** — name, age, gender, date of birth
- **Active conditions** — list of current health issues (with ICD-10 codes — standardized disease codes)
- **Medications** — what drugs the patient is currently taking
- **Allergies** — known drug/substance allergies
- **Lab results** — blood test values (with LOINC codes — standardized lab test codes)
- **Diagnostic reports** — imaging/test reports
- **Imaging available flag** — whether there's an X-ray to analyze

This model is the **contract between all agents**. If you change it, everything breaks.

### `shared/redis_client.py` — Caching

An async Redis wrapper. Used by Patient Context Agent (10-min cache for patient data) and Drug Safety Agent (1-hour cache for drug interaction results). Prevents hammering external APIs during development.

### `shared/sharp_context.py` — Platform Integration

Parses SHARP headers (a healthcare API protocol) from the Prompt Opinion platform. Allows the system to receive patient context from an external healthcare platform via HTTP headers.

### `knowledge_base/` — Medical RAG Data

Contains medical guideline text files (pneumonia treatment, drug interactions, lab references, differential diagnosis rules) that get chunked, embedded, and stored in ChromaDB. The `ingest.py` script is a one-time setup — it reads `.txt` files, splits them into chunks, generates vector embeddings via Google Gemini, and stores them in ChromaDB for the Diagnosis Agent and Consensus Agent to search against.

---

## The Agents — What Each One Does

### Agent 1: Patient Context Agent
**Port:** 8001 · **Endpoint:** `POST /fetch` · **Location:** `agents/patient_context/`

**One-liner:** Fetches a patient's complete medical record and normalizes it into a clean data object.

**What it actually does:**
1. Takes a patient ID as input
2. Makes **parallel async HTTP calls** to a FHIR server (a standardized health records API) to fetch 5 resource types simultaneously: Patient demographics, Conditions, Medications, Allergies, Lab Observations
3. Normalizes the messy FHIR JSON responses into a clean `PatientState` Pydantic model
4. Caches the result in Redis for 10 minutes (so repeated requests don't hit the external API)
5. Returns the normalized patient data to the caller

**Why it matters:** This is the **data foundation** — every other agent depends on its output. If this fails, nothing else runs. It handles missing data gracefully (returns empty arrays, doesn't crash).

**Key tech detail:** Uses `asyncio.gather()` to fetch all 5 FHIR resource types in parallel instead of sequentially — cuts response time from ~5s to ~1s.

---

### Agent 2: Diagnosis Agent
**Port:** 8002 · **Endpoint:** `POST /diagnose` · **Location:** `agents/diagnosis/`

**One-liner:** Figures out what's wrong with the patient using AI + medical guidelines.

**What it actually does:**
1. Takes the `PatientState` + a chief complaint (e.g., "fever and cough")
2. Searches ChromaDB for relevant medical guidelines using **RAG** (Retrieval-Augmented Generation) — finds the most relevant chunks from the knowledge base
3. Sends the patient data + retrieved guidelines to Google Gemini LLM
4. LLM returns a **ranked differential diagnosis** — a list of possible conditions sorted by likelihood, each with:
   - ICD-10 code (standardized disease code like `J18.9` for pneumonia)
   - Confidence score (0.0 to 1.0)
   - Clinical reasoning (why it thinks this)
   - Evidence for and against
5. Applies **rule-based adjustments** on top of the LLM output (e.g., if WBC and CRP labs are both elevated, bump up confidence for bacterial infection diagnoses)
6. Builds FHIR Condition resources (standardized output format)

**Why it matters:** This is the core "brain" of the system. All downstream agents react to what this agent says.

**Key tech detail:** Uses `with_structured_output()` on the LLM to force it to return a valid Pydantic model directly — no fragile JSON parsing needed.

---

### Agent 3: Lab Analysis Agent
**Port:** 8003 · **Endpoint:** `POST /analyze-labs` · **Location:** `agents/lab_analysis/`

**One-liner:** Analyzes blood test results to confirm or challenge the diagnosis.

**What it actually does — in two layers:**

**Layer 1 — Rules Engine (deterministic, always runs first):**
1. Loads `reference_ranges.json` — a lookup table of 16 lab tests with normal/abnormal/critical thresholds, broken down by age and gender
2. Classifies every lab result as `NORMAL`, `HIGH`, `LOW`, or `CRITICAL`
3. Runs **pattern detection** — checks for known combinations (e.g., high WBC + high CRP = "Bacterial infection markers")
4. Generates **critical alerts** that can never be overridden (e.g., "WBC 18.4 — extreme leukocytosis, sepsis workup mandatory")

**Layer 2 — LLM Interpretation (additive, never overrides Layer 1):**
5. Sends only the abnormal results to Google Gemini for qualitative interpretation
6. LLM says whether the lab pattern confirms or contradicts the Diagnosis Agent's conclusion
7. Outputs a `diagnosis_confirmation` object that the Consensus Agent reads

**Why it matters:** Provides an **independent second opinion** based purely on lab data. The Consensus Agent uses this to check if the Diagnosis Agent's conclusion holds up.

**Key design principle:** Critical alerts from the rules engine are **never suppressed** by the LLM. If a value is life-threateningly abnormal, it gets flagged no matter what.

---

### Agent 4: Drug Safety Agent (MCP Server)
**Port:** 8004 · **Endpoints:** `POST /check-safety` (REST) + `/mcp/` (MCP protocol) · **Location:** `agents/drug_safety/`

**One-liner:** Checks if proposed medications are safe for this specific patient.

**What it actually does:**
1. Takes a list of proposed drugs + patient's current meds, allergies, and conditions
2. Runs **three safety checks:**
   - **Allergy cross-reactivity** (local lookup) — e.g., patient is allergic to Penicillin → flag Amoxicillin because they're in the same drug family
   - **Condition contraindications** (local JSON database) — e.g., don't give Metformin to someone with severe kidney disease
   - **Drug-drug interactions** (external API call to NLM RxNav) — e.g., Amoxicillin + Warfarin = moderate interaction risk
3. Returns a verdict: `SAFE`, `CAUTION`, or `UNSAFE`
4. For cleared drugs, builds FHIR MedicationRequest resources
5. Can suggest **alternative medications** using LLM reasoning if a drug is flagged

**Dual-purpose architecture:** This agent serves as both:
- A **REST endpoint** for the MediTwin Orchestrator (`/check-safety`)
- A **standalone MCP tool server** (`/mcp/`) that any external agent on the Prompt Opinion Marketplace can call — three tools: `check_drug_interactions`, `get_contraindications`, `suggest_alternatives`

Both paths call the exact same core logic — there's one code path, not two.

**Key tech detail:** Uses FastMCP for the MCP protocol layer, mounted alongside FastAPI on the same port via Starlette routing.

---

### Agent 5: Imaging Triage Agent
**Port:** 8005 · **Endpoint:** `POST /analyze-xray` · **Location:** `agents/imaging_triage/`

**One-liner:** Analyzes chest X-ray images using a trained CNN to detect pneumonia.

**What it actually does:**
1. Accepts a **base64-encoded chest X-ray image** + patient context
2. Decodes the image, converts to RGB, resizes to 224×224 pixels
3. Applies exact same preprocessing as training (ImageNet normalization)
4. Runs inference through a **trained Keras CNN model** (binary classifier: NORMAL vs PNEUMONIA)
5. Classifies **severity** based on confidence + patient age:
   - Priority 1 (IMMEDIATE): confidence ≥ 90%
   - Priority 2 (URGENT): confidence ≥ 75%
   - Priority 3 (SEMI-URGENT): confidence ≥ 50%
   - Priority 4 (ROUTINE): below threshold
   - Elderly (>65) or pediatric (<5) patients get bumped up one priority level
6. Builds a FHIR DiagnosticReport with an **AI disclaimer** ("not a substitute for radiologist review")

**Conditional execution:** This agent only runs if `imaging_available = True` in the patient state. If no X-ray is provided, the system skips it entirely and still produces output.

**Graceful degradation:** If the `.h5` model file isn't present, the agent runs in "mock mode" — returns clearly-labeled placeholder output instead of crashing.

**Key tech detail:** TensorFlow inference runs in a `ThreadPoolExecutor` so it doesn't block the async event loop. Model is loaded once at startup and warmed up with a dummy prediction.

---

### Agent 6: Digital Twin Agent
**Port:** 8006 · **Endpoint:** `POST /simulate` · **Location:** `agents/digital_twin/`

**One-liner:** Creates a virtual copy of the patient and simulates "what would happen" under different treatment options.

**What it actually does:**
1. **Feature engineering** — extracts 19 ML features from PatientState (age, lab values, comorbidity flags, medication flags, etc.)
2. **Baseline risk prediction** — runs the feature vector through 3 pre-trained XGBoost models:
   - 30-day readmission risk
   - 30-day mortality risk  
   - Complication risk
3. **Treatment simulation** — for each treatment option (e.g., Option A: oral antibiotics, Option B: IV antibiotics, Option C: no treatment):
   - Applies **literature-based effect multipliers** (e.g., "Ceftriaxone IV reduces 30-day readmission by 55%")
   - Calculates modified risk scores and estimated recovery time
4. **Selects recommended option** using a weighted composite score (mortality 50%, readmission 30%, complications 20%)
5. **Feature attribution** — SHAP-style explanation showing which patient features contribute most to risk (e.g., "Age (72y) — increases risk", "CRP 145 (elevated) — increases risk")
6. **LLM narrative** — generates a 3-sentence clinical comparison constrained to cite specific numbers
7. **FHIR CarePlan** — packages the recommended treatment as a standard FHIR resource

**Model training:** `train_models.py` generates 2000 synthetic patients with clinically plausible distributions, trains 3 XGBoost classifiers, and saves them as `.json` files. This is a one-time setup step.

**Key design honesty:** The models are trained on synthetic data, not real hospital data. The architecture is production-correct, but the numbers are for demo purposes.

---

### Agent 7: Consensus + Escalation Agent
**Port:** 8007 · **Endpoint:** `POST /consensus` · **Location:** `agents/consensus/`

**One-liner:** The referee — checks if all agents agree, and raises a flag if they don't.

**What it actually does:**
1. Collects outputs from all specialist agents
2. Runs **conflict detection** (pure Python rules, no LLM) — checks for 3 types of disagreement:
   - **Diagnosis ↔ Lab mismatch** — Diagnosis Agent says one disease, Lab Agent's patterns suggest a different one (compared by ICD-10 code prefix)
   - **Imaging ↔ Clinical dissociation** — X-ray looks normal but clinical data strongly suggests pneumonia (possible early-stage disease not visible yet)
   - **Treatment contraindication** — Drug Safety Agent rejected the proposed medications
3. **Routes** based on conflict severity:
   - **No conflict** → `FULL_CONSENSUS` — all agents agree, proceed
   - **Moderate conflict** → attempt **Tiebreaker RAG** — queries the medical knowledge base to resolve the disagreement
   - **High severity conflict** → `ESCALATION_REQUIRED` — flags for human doctor review
4. Computes **weighted aggregate confidence** score:
   - Diagnosis Agent: 35% weight
   - Lab Analysis: 30% weight
   - Imaging: 25% weight
   - Tiebreaker bonus: 5% (for catching and fixing a conflict)
   - **Never exceeds 0.99** — the system is honest that it's never 100% certain

**Dual implementation:** Exists both as a REST endpoint (for HTTP calls) and as a **LangGraph node function** (for embedding inside the Orchestrator's state graph). Both call the same `run_consensus()` function.

---

### Agent 8: Explanation Agent
**Port:** 8007 · **Location:** (not yet built in the codebase)

**One-liner:** Translates all the technical JSON into human-readable reports.

**What it will do (per the strategy doc):**
1. **SOAP Note** for clinicians — structured medical note (Subjective, Objective, Assessment, Plan) in clinical terminology
2. **Patient Explanation** — same information but written at a 6th-grade reading level with no medical jargon. Uses `textstat` library to verify readability — if it's too complex, it regenerates
3. **Risk Attribution** — top 5 features driving the patient's risk, with plain-English labels ("Your elevated white blood cell count increases infection risk")
4. **FHIR Bundle** — assembles all FHIR resources from upstream agents (Condition, DiagnosticReport, MedicationRequest, CarePlan) into one standardized Bundle

---

### Agent 9: Orchestrator
**Port:** 8000 · **Location:** (not yet built in the codebase)

**One-liner:** The conductor — calls all other agents in the right order and manages the state.

**What it will do (per the strategy doc):**
1. Exposes the **single public API endpoint** — the only service the outside world talks to
2. Runs a **LangGraph StateGraph** that defines the execution flow:
   - Fetch patient context (must succeed — fatal if it fails)
   - Run Diagnosis + Lab Analysis **in parallel** (`asyncio.gather`)
   - Conditionally run Imaging (only if `imaging_available = True`)
   - Run Drug Safety + Digital Twin
   - Run Consensus
   - Run Explanation
3. **Error handling** — wraps every agent call in a `safe_agent_call` that catches timeouts and exceptions. Only Patient Context failure is fatal; all others degrade gracefully
4. **Health endpoint** — `/health` reports the status of all 8 downstream agents
5. **Target latency:** < 12 seconds with imaging, < 8 without

---

## Key Files Quick Reference

| File | What It Is |
|------|-----------|
| `docker-compose.yml` | Defines all 9+ services, ports, networks, and dependencies |
| `requirements.txt` | Python dependencies for the entire project |
| `shared/models.py` | Pydantic data models shared across all agents |
| `shared/redis_client.py` | Async Redis wrapper (caching) |
| `shared/sharp_context.py` | SHARP header parsing for platform integration |
| `agents/rag.py` | RAG chain for the Diagnosis Agent (ChromaDB + Gemini) |
| `agents/*/main.py` | FastAPI app for each agent |
| `agents/*/test.py` | Test scripts for each agent |
| `agents/lab_analysis/rules_engine.py` | Deterministic lab classification rules |
| `agents/lab_analysis/reference_ranges.json` | Lab test normal/abnormal thresholds |
| `agents/drug_safety/safety_core.py` | Cross-reactivity tables + contraindication logic |
| `agents/drug_safety/fda_client.py` | NLM RxNav + FDA API wrappers |
| `agents/drug_safety/contraindications.json` | Drug-condition contraindication database |
| `agents/imaging_triage/inference.py` | CNN model loading, preprocessing, inference |
| `agents/digital_twin/feature_engineering.py` | PatientState → ML feature vector |
| `agents/digital_twin/simulator.py` | Treatment effect multiplier engine |
| `agents/digital_twin/train_models.py` | XGBoost model training script |
| `agents/consensus/conflict_detector.py` | Rule-based conflict detection |
| `agents/consensus/tiebreaker.py` | RAG-based conflict resolution |
| `knowledge_base/ingest.py` | One-time ChromaDB seeding script |
| `knowledge_base/sources/*.txt` | Raw medical guideline documents |

---

## Port Map

| Port | Service |
|------|---------|
| 8000 | Orchestrator (main entry point) |
| 8001 | Patient Context Agent |
| 8002 | Diagnosis Agent |
| 8003 | Lab Analysis Agent |
| 8004 | Drug Safety MCP |
| 8005 | Imaging Triage Agent |
| 8006 | Digital Twin Agent |
| 8007 | Explanation Agent / Consensus Agent |
| 6379 | Redis |
| 8008 | ChromaDB |

---

## Build Order (Why It Matters)

The agents have strict dependency chains — you can't test downstream agents without upstream ones working:

```
1. Infrastructure (Redis, ChromaDB, FHIR sandbox)
2. Patient Context Agent (data foundation)
3. Diagnosis Agent + Lab Analysis Agent (can build in parallel)
4. Drug Safety MCP
5. Imaging Triage Agent
6. Digital Twin Agent
7. Consensus + Escalation
8. Explanation Agent
9. Orchestrator (LAST — ties everything together)
```

Building the Orchestrator first is a mistake — you'd be integrating against services that don't exist yet.

---

## Current Status

- ✅ Infrastructure (Docker Compose, Redis, ChromaDB, shared modules)
- ✅ Patient Context Agent (built, not fully tested)
- ✅ Diagnosis Agent (built with RAG pipeline)
- ✅ Lab Analysis Agent (rules engine + LLM layer)
- ✅ Drug Safety Agent (MCP + REST, with external API integration)
- ✅ Imaging Triage Agent (CNN inference pipeline, needs `.h5` model file)
- ✅ Digital Twin Agent (XGBoost + simulator + LLM narrative)
- ✅ Consensus Agent (conflict detection + tiebreaker RAG)
- 🔲 Explanation Agent (not yet built)
- 🔲 Orchestrator (not yet built)
