# MediTwin AI — Complete Backend Build Strategy

**Author:** Tayyab Hussain  
**Purpose:** Pre-code strategy document. Covers architecture decisions, sequencing, dependencies, risks, and agent-by-agent build plan before a single line of code is written.

---

## Part 1: Strategic Foundation

### What We're Actually Building

MediTwin is not a single application. It is a **distributed system of nine independently-running microservices**, each with its own process, its own dependencies, and its own failure surface. The Orchestrator is the only service the outside world talks to. Everything else is internal.

The mental model to keep throughout the build: think of it like nine separate mini-apps connected by HTTP, held together by a shared state object managed by LangGraph.

### The Three Things That Must Work Before Anything Else

Before any agent logic is written, three pieces of infrastructure must exist and be confirmed working:

1. **Redis** — if Redis is down, patient context caching fails, and the whole system slows to a crawl or breaks the Drug Safety agent's caching layer
2. **ChromaDB vector store** — if the medical knowledge base isn't indexed, the Diagnosis Agent and Consensus tiebreaker have nothing to query
3. **HAPI FHIR sandbox** — if we can't fetch a test patient from the public HAPI server, the Patient Context Agent cannot be validated end-to-end

These are not application code. They are prerequisites. Stand them up first, confirm they work, then start writing agents.

### Why This Order Matters

The agents have strict dependency chains. The safest build order is:

```
Infrastructure → Patient Context → (Diagnosis + Lab in parallel) → Drug Safety MCP → Digital Twin → Consensus → Explanation → Orchestrator (last)
```

The Orchestrator is built last because it is the integration layer — it only makes sense once all agents it calls actually exist. Building the Orchestrator first is the most common mistake in multi-agent system development. It leads to building an integration layer against nothing, which creates phantom bugs.

---

## Part 2: Infrastructure Strategy

### Docker Compose as the Single Source of Truth

Every service, including Redis, ChromaDB, and the HAPI FHIR mock, should be defined in a single `docker-compose.yml`. This means any developer (or judge) can run the entire system with one command.

Each agent runs as its own container on a dedicated port. The port map is:

- Orchestrator: 8000
- Patient Context Agent: 8001
- Diagnosis Agent: 8002
- Lab Analysis Agent: 8003
- Drug Safety MCP: 8004
- Imaging Triage Agent: 8005
- Digital Twin Agent: 8006
- Explanation Agent: 8007
- Redis: 6379
- ChromaDB: 8008 (internal only)

Containers should be on the same Docker network so they can call each other by service name, not by IP.

### Shared Infrastructure Components

**Redis** serves two agents: Patient Context (patient state cache, TTL 10 minutes) and Drug Safety (API response cache, TTL 1 hour). A single Redis instance is sufficient. It needs no special configuration — default Redis will work. The only thing that matters is that both agents point to the same Redis host.

**ChromaDB** is used by two agents: Diagnosis Agent (primary RAG queries) and Consensus Agent (tiebreaker RAG). They share the same vector store and the same ChromaDB instance. The collection name should be consistent — both agents must query the same collection or tiebreaker results will be incoherent.

**Medical Knowledge Base (ChromaDB ingestion)** is a one-time setup step that must happen before the Diagnosis Agent is usable. The ingest pipeline reads raw documents from the `knowledge_base/sources/` directory, chunks them, generates embeddings, and stores them in ChromaDB. This must be run as a separate setup script, not as part of agent startup. Sources: CDC ICD-10-CM descriptions, NIH MedlinePlus disease summaries, Synthea clinical notes.

**HAPI FHIR Sandbox** — for development, use the public HAPI FHIR R4 server at `https://hapi.fhir.org/baseR4`. Pre-load it with Synthea-generated synthetic patient data. The key thing to ensure is that the synthetic patients include all the resource types MediTwin needs: Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, DiagnosticReport. A Synthea export covers all of these by default.

### Shared Code Strategy

Create a `shared/` module that all agents import. This avoids duplicating:

- FHIR resource builders (the same Condition/MedicationRequest/CarePlan JSON structure is used across three agents)
- SHARP header parsing (every agent reads the same three headers)
- Redis client initialization (same connection config everywhere)
- Pydantic models for `PatientState` (the canonical shared data object)

The `PatientState` Pydantic model is the most critical shared piece. It is the contract between the Patient Context Agent and every downstream agent. If its schema changes, all agents break. Treat it as a versioned interface.

---

## Part 3: Agent Build Strategy

### Agent 1: Patient Context Agent

**What it is:** The data foundation. Every other agent depends on its output. If this agent fails, nothing else can run.

**Key strategic decisions:**

The FHIR fetch must be parallel using `asyncio.gather()`. Sequential fetches against the HAPI sandbox (six resource types) will take several seconds. Parallel reduces this to the time of the single slowest fetch.

Redis caching is non-negotiable. The Patient Context Agent will be called once per system invocation, but during development it will be called hundreds of times. Without caching, the HAPI sandbox will rate-limit the development process.

Missing FHIR resources must be handled gracefully. Not every synthetic patient has a DiagnosticReport. Not every patient has AllergyIntolerance entries. The normalization logic must return empty arrays for missing resource types, not raise exceptions.

**SHARP context strategy:** For the hackathon, the agent needs to work with both real SHARP headers (when called from Prompt Opinion) and a fallback input body (when called directly during development). Build a resolver that checks headers first, then falls back to request body. This means the development experience is smooth without needing to simulate SHARP headers every time.

**What to validate before moving on:** Make one full round-trip — call the Patient Context Agent with a Synthea patient ID, confirm it returns a populated `PatientState` with demographics, at least one condition, and at least one lab result, and confirm a second identical call hits Redis instead of FHIR.

---

### Agent 2: Diagnosis Agent

**What it is:** The clinical reasoning engine. Produces the ranked differential diagnosis list that all downstream agents either confirm or challenge.

**Key strategic decisions:**

The RAG pipeline has two phases: the one-time indexing phase and the per-request retrieval phase. These must be kept completely separate. The indexing pipeline should be a standalone script (`knowledge_base/ingest.py`) that is run once during setup. The agent itself only queries the already-built index.

Embedding model choice matters for latency. OpenAI `text-embedding-3-small` is fast and cheap and sufficient for this use case. Don't use a large embedding model for a hackathon demo.

The LLM prompt for differential diagnosis must enforce JSON output strictly. Use a structured output instruction at the system level, not just in the user prompt. The output parser will fail unpredictably if the LLM adds preamble text before the JSON.

Confidence scores from the LLM are not reliable by themselves. The rule-based post-processing step (boost confidence if labs confirm, reduce if allergy conflict with first-line treatment) adds determinism on top of LLM uncertainty. This is what makes the confidence number defensible.

**ICD-10 code discipline:** Every diagnosis in the output must carry a valid ICD-10 code. The downstream Consensus Agent compares codes, not display names. If the Diagnosis Agent says "Pneumonia" and the Lab Agent says "J18.9", the Consensus Agent will flag a mismatch when there shouldn't be one. Enforce ICD-10 in the output schema.

**What to validate before moving on:** Feed in a PatientState representing a patient with fever, cough, elevated WBC. Confirm the agent returns J18.9 or J22 as top diagnosis with a confidence above 0.7, and that the output includes `recommended_next_steps` (needed by the Drug Safety Agent downstream).

---

### Agent 3: Lab Analysis Agent

**What it is:** The pathology layer. Provides independent confirmation or contradiction of the Diagnosis Agent's conclusion based purely on lab values.

**Key strategic decisions:**

The rules engine and the LLM layer must be strictly separated. The rules engine runs first and is entirely deterministic — it flags CRITICAL values based on hardcoded thresholds. The LLM only runs afterward to interpret patterns. This means a critical WBC of 18.4 will always be flagged as CRITICAL regardless of what the LLM says.

The LOINC code lookup is the foundation of this agent. Build a `reference_ranges.json` that maps LOINC codes to their clinical reference ranges, segmented by age group and gender. The minimum viable set is the ten codes listed in the agent spec: WBC, Hemoglobin, Platelets, Creatinine, ALT, Glucose, Albumin, Potassium, Troponin, CRP. Everything else can fail gracefully.

The `diagnosis_confirmation` output field is the critical interface point with the Consensus Agent. It must contain a boolean `confirms_top_diagnosis` and the ICD-10 code being confirmed or denied. Without this field, the Consensus Agent's conflict detection logic cannot function.

**Age-adjusted reference ranges** matter for the demo. A WBC of 14.2 in an adult is HIGH. In a child under 5 it is borderline normal. If the demo patient is an adult, this is fine. But build age adjustment into the lookup from the start — it's straightforward to implement and makes the system more credible.

**What to validate before moving on:** Feed in a PatientState with WBC of 18.4. Confirm the agent flags it as CRITICAL, identifies the bacterial infection pattern, and sets `confirms_top_diagnosis: true` for J18.9.

---

### Agent 4: Drug Safety Agent (MCP Superpower)

**What it is:** The medication safety gate and the standalone Superpower submission. This is the dual-submission advantage.

**Key strategic decisions:**

This agent serves two purposes simultaneously: as a component inside MediTwin and as a standalone MCP tool on the Prompt Opinion Marketplace. The MCP server interface must be built first, and the MediTwin integration must consume it the same way any external agent would. Do not build two separate code paths — build the MCP server and then have MediTwin call it like any other MCP consumer.

The three external APIs (FDA OpenFDA, NLM RxNav, NLM RxNorm) are all free and require no authentication. However, they are external services and will have latency. All three calls should be made asynchronously. Redis caching with a 1-hour TTL is essential here — the same drug combination will be checked multiple times during development.

The cross-reactivity lookup table and the condition contraindications database are local rules that must be hardcoded and reviewed by a clinician (or at minimum, sourced from a credible clinical reference). These are not generated by the LLM. The LLM is only used for the `suggest_alternatives` tool, where clinical reasoning about equivalent drug classes is needed.

The `safety_status` field in the output is binary: SAFE or UNSAFE. The Digital Twin Agent and the Consensus Agent both key off this field. Make sure UNSAFE is returned whenever there is any CRITICAL interaction or any HIGH-severity contraindication.

**FHIR MedicationRequest output:** Every drug that passes safety checks should be returned as a FHIR MedicationRequest resource. This is what the Explanation Agent's FHIR Bundle assembly will use.

**MCP tool schema discipline:** The three tool schemas (`check_drug_interactions`, `get_contraindications`, `suggest_alternatives`) are the public API of the Superpower submission. They must be clean, well-documented, and match the JSON Schema the MCP SDK expects. The `patient_id` parameter on `check_drug_interactions` is optional and enables SHARP context passthrough — implement this but make sure the tool works without it.

**What to validate before moving on:** Call `check_drug_interactions` with ["Amoxicillin", "Warfarin"]. Confirm it returns a MODERATE interaction. Then call `get_contraindications` for Amoxicillin with allergy "Penicillin". Confirm it returns a HIGH contraindication. Then call `suggest_alternatives` and confirm Azithromycin appears.

---

### Agent 5: Imaging Triage Agent

**What it is:** The biggest technical differentiator. A trained CNN model wrapped as a FHIR-compliant A2A service.

**Key strategic decisions:**

The model loading strategy is critical. The CNN model (`.h5` file) must be loaded once at application startup into `app.state.model`, not on every request. A Keras model load takes several seconds. Loading it per-request would make the agent unusable.

Preprocessing must exactly match the training pipeline. The Kaggle pneumonia dataset was preprocessed with specific normalization values. If the inference preprocessing differs from training preprocessing, the model's accuracy collapses. The preprocessing pipeline must be copied from the training notebook, not reimplemented from memory.

The severity classification layer on top of the model output is pure business logic — it maps pneumonia probability to a triage priority. This layer is not ML. It is hardcoded thresholds with age-based adjustments. Keep it simple and transparent.

The FHIR DiagnosticReport output must include the AI disclaimer in the `conclusion` field. This is not optional — it's the difference between a clinical decision support tool and something that could be mistaken for a radiologist report.

**Conditional execution in the LangGraph:** The Imaging Agent only runs if `state["imaging_available"]` is true. The Orchestrator graph has a conditional edge that routes around this agent if no image is provided. This means the agent must never be required for the system to produce output.

**What to validate before moving on:** Submit a sample chest X-ray from the Kaggle dataset (one from the PNEUMONIA folder). Confirm the agent returns prediction "PNEUMONIA", confidence above 0.85, triage_priority of 1 or 2, and a valid FHIR DiagnosticReport JSON structure.

---

### Agent 6: Digital Twin Agent

**Key strategic decisions:**

The XGBoost models must be trained before this agent is written. Training requires the MIMIC-III Demo dataset. Download it from PhysioNet, run the feature engineering pipeline, and serialize the trained models to `.json` files before building the agent. The agent at runtime only loads and runs inference — it does not train.

Be honest about the model's limitations in the architecture documentation. The MIMIC-III demo has 100 patients. The models will produce plausible-looking outputs but are not clinically validated. Frame this correctly: "demonstrates the architecture of a digital twin, trained on a small demo dataset."

The treatment effect modifiers are the most clinically sensitive part of this agent. The reduction factors (e.g., "Ceftriaxone IV reduces 30-day readmission risk by 62%") are literature-based heuristics. Source these from actual published studies if possible, or cite the basis for the numbers. This is the part of the demo most likely to be scrutinized by a medically-informed judge.

The `what_if_narrative` is generated by LLM, but it should be tightly constrained. The prompt should specify that the narrative must be three sentences, must cite specific numbers from the simulation, and must not introduce clinical claims beyond what the scenario data supports.

**FHIR CarePlan output:** The recommended treatment scenario should be returned as a FHIR CarePlan resource. The activity array should list each drug as a separate activity with `kind: MedicationRequest`.

**What to validate before moving on:** Feed in a PatientState with a 54-year-old male, WBC 18.4, diagnosis J18.9. Confirm three scenarios are returned, that Option B has higher recovery probability than Option A, and that Option C (no treatment) shows the highest mortality risk.

---

### Agent 7: Consensus + Escalation Agent

**What it is:** A LangGraph conditional node, not a standalone microservice. Lives inside the Orchestrator's graph.

**Key strategic decisions:**

This is the most architecturally nuanced part of the system. It is not an HTTP service — it is a function that runs inside the LangGraph graph. The conflict detection logic is a pure Python rules engine, and the tiebreaker is a RAG query using the same ChromaDB index as the Diagnosis Agent.

The three conflict types to implement are: diagnosis-lab mismatch (top ICD-10 codes differ), imaging-clinical dissociation (imaging says normal but clinical suspicion is high), and treatment contraindication (Drug Safety says UNSAFE for the proposed treatment). Each has its own conflict severity level.

The routing logic is conservative: when in doubt, escalate. A false escalation (calling for human review unnecessarily) is safe. A missed conflict in a clinical system is not. Set the severity threshold for escalation low.

The aggregate confidence scoring must be weighted correctly. The Diagnosis Agent's confidence gets the highest weight (0.35) because it synthesizes the most information. Lab confirmation (0.30) is the second most reliable signal. Imaging (0.25) is weighted slightly lower because the CNN model has known limitations. Never let the aggregate reach 1.0 — cap at 0.99.

**LangGraph state flow:** The Consensus Agent writes its output to the shared `MediTwinState`. The Explanation Agent then reads from that state. The state fields that matter most are `consensus_status` (FULL_CONSENSUS, CONFLICT_RESOLVED, or ESCALATION_REQUIRED), `human_review_required` (boolean), and `final_assessment` (the merged clinical picture).

**What to validate before moving on:** Test three scenarios manually: (1) all agents agree — confirm FULL_CONSENSUS, (2) Diagnosis says viral pneumonia but Lab shows bacterial markers — confirm CONFLICT_RESOLVED with tiebreaker RAG, (3) imaging is normal but clinical suspicion is high — confirm ESCALATION_REQUIRED.

---

### Agent 8: Explanation Agent

**What it is:** The last mile. Transforms technical JSON into a SOAP note, a patient explanation, and a FHIR Bundle.

**Key strategic decisions:**

Two separate LLM prompts must be used: one for the clinician SOAP note (clinical terminology, structured format) and one for the patient explanation (grade 6 reading level, no jargon). These cannot share a prompt. The tone, vocabulary, and output structure are completely different.

The reading level check using `textstat` should be a hard gate, not a suggestion. If the Flesch-Kincaid grade level exceeds 8, regenerate the patient explanation with a stricter prompt. Build a retry loop with a maximum of two retries before accepting the output with a warning flag.

The SHAP-style risk attribution is not actual SHAP. It uses XGBoost's `get_score()` method with `importance_type='gain'` to produce feature importance values, then maps those to human-readable sentences with directional framing (increases risk / reduces risk). The output should show the top 5 contributing features.

**FHIR Bundle assembly** is the final technical step. The bundle collects all FHIR resources generated by upstream agents: the Condition from the Diagnosis Agent, the DiagnosticReport from the Imaging Agent, the MedicationRequests from the Drug Safety Agent, and the CarePlan from the Digital Twin Agent. The Explanation Agent assembles these into a single Bundle resource with a unique ID and timestamp.

The medical disclaimer must appear in both the SOAP note and the patient explanation. Build it in as a template string that is always appended, not something that depends on the LLM to include.

**What to validate before moving on:** Feed in a complete multi-agent output. Confirm the SOAP note has all four sections (S/O/A/P), the patient explanation is at grade 6-8 reading level, the FHIR Bundle contains at least four resource entries, and the risk attribution shows at least three features.

---

### Agent 9: Orchestrator

**What it is:** The central nervous system. Built last because it requires all other agents to exist.

**Key strategic decisions:**

The LangGraph `StateGraph` is the core of this agent. Define the full graph — all nodes and all edges including conditional edges — before implementing any of the node functions. Draw it out first, then code it. The graph structure is the contract between all agents.

The parallel execution node for Diagnosis and Lab is implemented with `asyncio.gather()`. Both HTTP calls happen simultaneously. The Orchestrator waits for both to complete before proceeding to imaging. This is the primary latency optimization.

The error handling strategy is: no single agent failure should crash the system. Only Patient Context failure is fatal. All other agents use a `safe_agent_call` wrapper that catches timeouts and exceptions, logs them to `error_log` in the state, and returns `None`. Downstream agents that receive `None` must handle it gracefully.

The A2A Agent Card registration on Prompt Opinion is a manual step, not automated code. Prepare the JSON metadata object and register it via the platform's UI. The input schema and SHARP context declaration in the Agent Card must exactly match what the Orchestrator's FastAPI endpoint accepts.

**Execution timeline target:** The goal is under 12 seconds for a full analysis with imaging, under 8 without. The biggest latency risks are the LLM calls in the Diagnosis and Explanation agents. Use GPT-4o-mini or Claude Haiku (not GPT-4 or Claude Opus) to stay within the latency budget. Pre-warm the CNN model and XGBoost models at startup.

**Health check endpoint:** Expose a `/health` endpoint that reports the status of all downstream agents. This is useful during the demo to show that all eight services are running.

---

## Part 4: Data Flow Contract

### The PatientState Object

`PatientState` is the most critical shared data structure in the system. It is produced by Agent 1 and consumed by every downstream agent. Its schema must be finalized before any agent is built. Changing it later cascades to all consumers.

Minimum required fields:
- `patient_id` (string)
- `demographics` (object: name, age, gender, dob)
- `active_conditions` (array of objects: code, display, onset)
- `medications` (array of objects: drug, dose, frequency, status)
- `allergies` (array of objects: substance, reaction, severity)
- `lab_results` (array of objects: loinc, display, value, unit, flag)
- `diagnostic_reports` (array of objects)
- `recent_encounters` (array of objects)
- `state_timestamp` (ISO 8601 string)
- `imaging_available` (boolean — derived from whether DiagnosticReport with imaging content exists)

### The MediTwinState Object (LangGraph)

This is the graph-level state object. It carries all agent outputs and control flags through the graph. Key discipline: each node only writes to its own designated fields. No node should write to another node's output fields.

### The Inter-Agent HTTP Contract

Each agent exposes exactly one primary endpoint (e.g., `/diagnose`, `/analyze-labs`, `/analyze-xray`). The request and response schemas for each endpoint must be documented before the endpoint is implemented. These are internal APIs but treat them as if they are public — any change to them requires updating the Orchestrator's call code.

---

## Part 5: Testing Strategy

### Unit Test Priority

Test these in isolation before integration:

1. Reference range classification in the Lab Analysis Agent (deterministic, easy to unit test)
2. Cross-reactivity lookup in the Drug Safety Agent (hardcoded table, should be 100% reliable)
3. Conflict detection rules in the Consensus Agent (three rule types, test each independently)
4. CNN preprocessing pipeline (compare output against known-good values from training)
5. Feature engineering in the Digital Twin Agent (verify all LOINC lookups return expected values)

### Integration Test Scenarios

Test with three synthetic patients that cover different clinical paths:

**Patient A (Happy Path):** Adult male with CAP symptoms, elevated WBC, chest X-ray positive. Expected result: FULL_CONSENSUS, J18.9, IV antibiotic recommendation, no drug safety alerts.

**Patient B (Drug Safety Alert):** Same as Patient A but with documented Penicillin allergy and current Warfarin prescription. Expected result: Drug Safety flags Amoxicillin contraindication, suggests Azithromycin, Warfarin interaction noted.

**Patient C (Escalation Path):** Patient with high clinical suspicion for pneumonia but normal chest X-ray (imaging-clinical dissociation). Expected result: ESCALATION_REQUIRED, human review flag, partial outputs still available.

### The Demo Smoke Test

Before recording the demo video, run a full end-to-end pass with timing. Confirm:
- Total time under 12 seconds (with imaging)
- All eight agent outputs present in final state
- FHIR Bundle contains correct resource types
- Patient explanation reads at grade 6-8 level
- No unhandled exceptions in any agent log

---

## Part 6: Risk Register and Mitigations

### Risk 1: FHIR Sandbox Instability
The public HAPI FHIR sandbox goes down occasionally. Mitigation: pre-cache a JSON snapshot of the demo patient's complete PatientState. If the HAPI call fails, fall back to the snapshot for demo purposes.

### Risk 2: LLM API Rate Limiting
During the demo, back-to-back LLM calls across five agents could hit rate limits. Mitigation: use the same API key but stagger calls through the LangGraph sequential execution. Add retry logic with exponential backoff on 429 responses.

### Risk 3: CNN Model Loading Time
The Keras model takes several seconds to load. If the Imaging Agent container restarts mid-demo, the first request will be slow. Mitigation: implement a startup health check that pre-warms the model. The orchestrator's health endpoint can verify the imaging agent has its model loaded.

### Risk 4: ChromaDB Cold Start
If ChromaDB is not pre-seeded with the medical knowledge base, the Diagnosis Agent returns empty RAG context and falls back to pure LLM reasoning without grounding. Mitigation: include the ChromaDB seed data in the repository and run the ingest script as a Docker Compose `init` container that runs before the agents start.

### Risk 5: LangGraph State Serialization
LangGraph state must be serializable. All values in `MediTwinState` must be JSON-serializable types. NumPy arrays from the Digital Twin must be converted to Python lists before being stored in state.

### Risk 6: A2A Platform Integration Timing
The Prompt Opinion platform integration may have unexpected requirements discovered late. Mitigation: start the A2A registration in week 1. Test the SHARP header passthrough immediately. Do not leave platform integration to the last day.

---

## Part 7: Build Sequence Summary

The correct order to build the system, with no step depending on a step that comes after it:

1. Set up Docker Compose with Redis, ChromaDB, and network configuration
2. Run Synthea and pre-load HAPI FHIR sandbox with synthetic patient data
3. Run ChromaDB ingestion pipeline with medical knowledge base documents
4. Train XGBoost risk models on MIMIC-III demo data and serialize them
5. Build and validate Patient Context Agent (Port 8001)
6. Build and validate Diagnosis Agent (Port 8002)
7. Build and validate Lab Analysis Agent (Port 8003) in parallel with step 6
8. Build and validate Drug Safety MCP Server (Port 8004) — publish to Marketplace
9. Build and validate Imaging Triage Agent (Port 8005)
10. Build and validate Digital Twin Agent (Port 8006)
11. Build Consensus + Escalation logic (LangGraph node, lives inside Orchestrator module)
12. Build and validate Explanation Agent (Port 8007)
13. Build Orchestrator with LangGraph graph, connecting all agents (Port 8000)
14. End-to-end integration test with all three synthetic patient scenarios
15. Register MediTwin as A2A agent on Prompt Opinion Platform
16. Record demo video
17. Final submission checklist validation

---

## Part 8: What Makes This Win

From the judging criteria:

**Standards compliance** is achieved by: FHIR R4 I/O across all agents, SHARP context propagation through the Orchestrator to Patient Context Agent, A2A registration on the platform, and MCP server published independently.

**Technical depth** is demonstrated by: eight running microservices, LangGraph graph orchestration with conditional routing, a trained CNN model (not a pre-trained API), XGBoost risk models trained on real clinical data, and a full RAG pipeline over a medical knowledge base.

**Differentiation** comes from: the trained imaging model (no other team will have this), the digital twin simulation with what-if scenarios (rare in any clinical AI demo), and the dual-submission strategy where the Drug Safety MCP is independently useful.

**Healthcare relevance** is shown by: FHIR R4 compliance throughout, LOINC and ICD-10 code discipline, clinical reference range databases, FDA and NLM API integration for drug safety, and honest framing of AI limitations in all clinical outputs.

The one-sentence pitch to keep in mind throughout the build: **"What is happening? What will happen next? What should we do?"** — every architectural decision should serve those three questions.