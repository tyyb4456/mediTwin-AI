# MediTwin AI — System Architecture & Workflow

**Project Name:** MediTwin AI  
**Hackathon:** Agents Assemble — Healthcare AI Endgame Challenge  
**Submission Tracks:** Full Agent (A2A) + Superpower (MCP)  
**Author:** Tayyab Hussain  
**Tagline:** *"What is happening? What will happen next? What should we do?"*

---

## 1. System Overview

MediTwin AI is a multi-agent clinical decision support system that combines eight specialist AI agents to produce a comprehensive medical assessment from a single patient input. It replaces the "AI chatbot" model with a structure that mirrors how real clinical teams work — each specialist contributes their domain expertise, disagreements are caught and resolved, and the final output is tailored for both clinicians and patients.

The system operates on two parallel tracks for the hackathon:

- **Superpower submission (MCP):** The Drug Safety Agent is published as a standalone MCP server on the Prompt Opinion Marketplace. Any agent on the platform can discover and invoke its tools (`check_drug_interactions`, `get_contraindications`, `suggest_alternatives`).

- **Full Agent submission (A2A):** MediTwin as a complete system is registered as an A2A agent. It accepts SHARP context from the Prompt Opinion platform and returns a structured clinical output.

---

## 2. The Core Insight

Most AI healthcare demos answer one question: **"What is happening?"**

MediTwin answers three:

| Question | Agent responsible | Output |
|---|---|---|
| What is happening? | Diagnosis + Lab + Imaging agents | Differential diagnosis, 91% confidence |
| What will happen next? | Digital Twin agent | 30-day readmission risk, treatment response curves |
| What should we do? | Drug Safety + Orchestrator + Consensus | Safe treatment plan, FHIR CarePlan |

---

## 3. Agent Directory

| # | Agent | Type | Role |
|---|---|---|---|
| 1 | Patient Context Agent | A2A | FHIR data ingestion — system entry point |
| 2 | Diagnosis Agent | A2A | RAG-based differential diagnosis |
| 3 | Lab Analysis Agent | A2A | Abnormality detection + pattern interpretation |
| 4 | Drug Safety Agent | **MCP Server** | Drug interactions + contraindications |
| 5 | Imaging Triage Agent | A2A | CNN-based chest X-ray analysis |
| 6 | Digital Twin Agent | A2A | Outcome simulation — what-if scenarios |
| 7 | Consensus + Escalation Agent | A2A (LangGraph node) | Conflict detection + arbitration |
| 8 | Explanation Agent | A2A | SOAP note + patient communication |
| 9 | Orchestrator Agent | A2A (Entry point) | Graph sequencing + state management |

---

## 4. Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT OPINION PLATFORM                       │
│            (provides SHARP context + A2A routing)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ A2A call + SHARP headers
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR AGENT (Port 8000)                  │
│                    LangGraph StateGraph                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. PATIENT CONTEXT AGENT                               │   │
│  │  Fetches FHIR R4 resources → builds PatientState        │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │ PatientState                         │
│            ┌──────────────┴──────────────┐                     │
│            ▼                             ▼                      │
│  ┌─────────────────┐          ┌──────────────────────┐         │
│  │ 2. DIAGNOSIS    │          │ 3. LAB ANALYSIS      │         │
│  │ AGENT           │          │ AGENT                │         │
│  │ RAG + LLM       │          │ Rules + LLM          │         │
│  │ → Differential  │          │ → Abnormality flags  │         │
│  └────────┬────────┘          └──────────┬───────────┘         │
│           │                              │                      │
│           └──────────────┬───────────────┘                     │
│                          │ (if image provided)                  │
│                          ▼                                      │
│               ┌────────────────────┐                           │
│               │ 5. IMAGING TRIAGE  │                           │
│               │ AGENT              │                           │
│               │ CNN inference      │                           │
│               │ → DiagnosticReport │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│                         ▼                                       │
│               ┌────────────────────┐                           │
│               │ 4. DRUG SAFETY     │◄──── Also exposed as      │
│               │ MCP SERVER         │      standalone MCP        │
│               │ FDA API + rules    │      on Marketplace        │
│               │ → Safety verdict   │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│                         ▼                                       │
│               ┌────────────────────┐                           │
│               │ 6. DIGITAL TWIN    │                           │
│               │ AGENT              │                           │
│               │ XGBoost risk model │                           │
│               │ → Scenario compare │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│                         ▼                                       │
│               ┌────────────────────┐                           │
│               │ 7. CONSENSUS +     │                           │
│               │ ESCALATION AGENT   │                           │
│               │ Conflict detection │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│              ┌──────────┴──────────┐                           │
│              ▼                     ▼                            │
│        [consensus]          [escalate to human]                 │
│              │                     │                            │
│              └──────────┬──────────┘                           │
│                         ▼                                       │
│               ┌────────────────────┐                           │
│               │ 8. EXPLANATION     │                           │
│               │ AGENT              │                           │
│               │ SOAP note + FHIR   │                           │
│               │ Bundle + patient   │                           │
│               │ communication      │                           │
│               └─────────┬──────────┘                           │
│  ┌──────────────────────▼────────────────────────────────┐    │
│  │                  SHARED INFRASTRUCTURE                  │    │
│  │  Redis cache │ ChromaDB vector store │ HAPI FHIR server │    │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Final MediTwin output
                             ▼
              FHIR Bundle + SOAP Note + Patient Summary
              + Risk Scores + Treatment Plan + Drug Safety Report
```

---

## 5. Data Flow — Step by Step

### Step 1: Request Arrives at Orchestrator
- Prompt Opinion platform sends A2A request with SHARP headers
- SHARP headers contain: `X-SHARP-Patient-ID`, `X-SHARP-FHIR-Token`, `X-SHARP-FHIR-BaseURL`
- Orchestrator initializes LangGraph state

### Step 2: Patient Context Agent (Sequential — others depend on this)
- Reads SHARP token from state
- Makes parallel async FHIR R4 calls: Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, DiagnosticReport
- Normalizes into `PatientState` Pydantic model
- Caches in Redis (TTL: 10 min)
- Returns: Complete structured patient state

### Step 3: Diagnosis + Lab Agents (Parallel — no dependency between them)
- Both receive `PatientState` simultaneously via `asyncio.gather()`
- Diagnosis Agent runs RAG over medical knowledge base
- Lab Analysis Agent runs rules engine over Observation resources
- Both complete independently
- Returns: Differential diagnosis list + annotated lab report

### Step 4: Imaging Triage Agent (Conditional — only if X-ray provided)
- LangGraph conditional edge: if `state["imaging_available"]` → route to imaging
- Receives base64 X-ray + patient context
- Runs trained pneumonia CNN
- Returns: FHIR DiagnosticReport + triage priority

### Step 5: Drug Safety Agent (Sequential — needs proposed treatment)
- Receives current medications + allergies from PatientState
- Receives proposed treatment from Diagnosis Agent's "recommended next steps"
- Calls FDA OpenFDA and RxNav APIs
- Returns: Interaction flags + contraindications + safe alternatives

### Step 6: Digital Twin Agent (Sequential — needs all above)
- Receives full patient state + all agent outputs
- Engineers ML features from FHIR data
- Runs XGBoost risk models
- Simulates treatment scenarios (A vs B vs no treatment)
- Returns: Scenario comparison + FHIR CarePlan

### Step 7: Consensus + Escalation Agent (Sequential — reviews all outputs)
- Compares Diagnosis Agent vs Lab Agent: do they agree on top diagnosis?
- Compares Imaging Agent vs Clinical: imaging-clinical dissociation?
- Checks for unresolvable Drug Safety conflicts
- Routes: `full_consensus` → explanation | `conflict` → tiebreaker → explanation | `escalation` → human review flag
- Returns: Consensus status + aggregate confidence

### Step 8: Explanation Agent (Sequential — final step)
- Receives all agent outputs + consensus verdict
- Generates SOAP note for clinician
- Generates plain-language summary for patient (Grade 6 reading level)
- Applies SHAP-style risk attribution
- Assembles FHIR Bundle of all generated resources
- Returns: Complete MediTwin final output

---

## 6. FHIR Resources — Input and Output

### Input Resources (fetched from FHIR server)
| Resource | Purpose | Key fields used |
|---|---|---|
| `Patient` | Demographics | id, birthDate, gender, name |
| `Condition` | Diagnoses | code (ICD-10), clinicalStatus, onsetDateTime |
| `MedicationRequest` | Prescriptions | medicationCodeableConcept, dosageInstruction, status |
| `AllergyIntolerance` | Drug allergies | code, reaction, severity |
| `Observation` | Labs + vitals | code (LOINC), valueQuantity, referenceRange |
| `DiagnosticReport` | Imaging reports | code, conclusion, presentedForm |

### Output Resources (generated by MediTwin)
| Resource | Generated by | Content |
|---|---|---|
| `Condition` | Diagnosis Agent | AI-generated differential (verificationStatus: provisional) |
| `DiagnosticReport` | Imaging Triage Agent | CNN triage result + confidence score |
| `MedicationRequest` | Drug Safety Agent | Approved medications with safety clearance |
| `CarePlan` | Digital Twin Agent | Recommended treatment plan (highest probability scenario) |
| `Bundle` | Explanation Agent | Collection of all above resources |

---

## 7. SHARP Context Integration

SHARP (Standardized Healthcare API Request Protocol) is the Prompt Opinion platform's mechanism for propagating EHR session credentials through multi-agent chains.

MediTwin consumes SHARP context at the Orchestrator level and passes it to the Patient Context Agent:

```python
# Orchestrator reads SHARP headers
patient_id = request.headers.get("X-SHARP-Patient-ID")
fhir_token = request.headers.get("X-SHARP-FHIR-Token")
fhir_base_url = request.headers.get("X-SHARP-FHIR-BaseURL")

# Patient Context Agent uses them for FHIR requests
headers = {"Authorization": fhir_token}
response = await httpx.get(f"{fhir_base_url}/Patient/{patient_id}", headers=headers)
```

The Drug Safety MCP Server also exposes an optional `patient_id` parameter — when provided, it uses SHARP context to pull the patient's current medication list automatically instead of requiring manual input.

---

## 8. Technology Stack Summary

| Layer | Technology | Used by |
|---|---|---|
| Graph orchestration | LangGraph | Orchestrator |
| Agent communication | A2A + HTTP | All agents |
| MCP server | FastAPI + MCP SDK | Drug Safety Agent |
| LLM reasoning | GPT-4o-mini / Claude Haiku | Diagnosis, Lab, Explanation |
| RAG pipeline | LangChain + ChromaDB | Diagnosis, Consensus |
| Medical imaging | TensorFlow/Keras CNN | Imaging Triage |
| ML risk models | XGBoost | Digital Twin |
| FHIR client | httpx (async) + fhirclient | Patient Context |
| Caching | Redis | Patient Context, Drug Safety |
| Data validation | Pydantic v2 | All agents |
| API framework | FastAPI | All agents |
| FHIR server (dev) | HAPI FHIR public sandbox | Patient Context |
| Deployment | Docker Compose | All services |

---

## 9. Project File Structure

```
meditwin/
│
├── orchestrator/
│   ├── main.py              # FastAPI entry point + A2A registration
│   ├── graph.py             # LangGraph StateGraph definition
│   ├── state.py             # MediTwinState TypedDict
│   └── router.py            # Consensus routing logic
│
├── agents/
│   ├── patient_context/
│   │   ├── main.py          # FastAPI service
│   │   ├── fhir_client.py   # Async FHIR fetcher
│   │   └── models.py        # PatientState Pydantic models
│   │
│   ├── diagnosis/
│   │   ├── main.py
│   │   ├── rag_chain.py     # LangChain RAG setup
│   │   └── knowledge_base/  # ChromaDB index
│   │
│   ├── lab_analysis/
│   │   ├── main.py
│   │   ├── rules_engine.py  # Abnormality detection rules
│   │   └── reference_ranges.json
│   │
│   ├── drug_safety/         # Also the MCP Superpower
│   │   ├── main.py
│   │   ├── mcp_server.py    # MCP tool definitions
│   │   ├── fda_client.py    # OpenFDA API wrapper
│   │   └── contraindications.json
│   │
│   ├── imaging_triage/
│   │   ├── main.py
│   │   ├── inference.py     # CNN model loader + preprocessor
│   │   └── models/
│   │       └── pneumonia_cnn_v1.h5
│   │
│   ├── digital_twin/
│   │   ├── main.py
│   │   ├── feature_engineering.py
│   │   ├── simulator.py     # Treatment effect modifiers
│   │   └── models/
│   │       ├── readmission_30d.json
│   │       └── mortality_30d.json
│   │
│   ├── consensus/
│   │   ├── conflict_detector.py
│   │   └── tiebreaker.py
│   │
│   └── explanation/
│       ├── main.py
│       ├── soap_generator.py
│       ├── patient_writer.py
│       └── fhir_bundler.py
│
├── shared/
│   ├── fhir_models.py       # Shared FHIR resource builders
│   ├── redis_client.py      # Shared Redis connection
│   └── sharp_context.py     # SHARP header parsing
│
├── knowledge_base/
│   ├── ingest.py            # Indexing pipeline
│   └── sources/             # Raw medical documents
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 10. Demo Video Script (3 Minutes)

### Minute 1: The Problem + Entry Point
- Show the Prompt Opinion platform
- Input a patient ID (use HAPI FHIR synthetic patient)
- Show SHARP context being injected
- Show MediTwin activating as an A2A agent
- Narrate: "Traditional AI gives you one answer. MediTwin assembles a team."

### Minute 2: The Agents Working
- Show the orchestrator graph executing (LangGraph visualization)
- Highlight the parallel execution of Diagnosis + Lab agents
- Show the chest X-ray being submitted and the CNN returning 92.3% pneumonia confidence
- Show the Consensus Agent detecting agreement across all three sources
- Show the Digital Twin comparing three treatment scenarios with probability curves
- Show the Drug Safety MCP flagging the Penicillin allergy

### Minute 3: The Output + Marketplace
- Show the final SOAP note
- Show the patient-friendly explanation
- Show the FHIR Bundle being generated
- Navigate to the Prompt Opinion Marketplace and show the Drug Safety MCP published independently
- Demonstrate another agent calling the Drug Safety MCP tool
- Closing: "One system. Eight specialists. Three questions answered."

---

## 11. Judging Criteria Alignment

| Criterion | How MediTwin addresses it |
|---|---|
| Standards compliance (MCP/A2A/FHIR) | Full FHIR R4 I/O, SHARP context, A2A registration, MCP server published |
| Healthcare relevance | Addresses real clinical workflow — diagnosis, drug safety, imaging triage |
| Technical depth | 8 agents, LangGraph graph, XGBoost risk model, trained CNN, RAG pipeline |
| Differentiation | Only submission with a trained medical imaging model; digital twin simulation |
| Platform integration | Drug Safety MCP is independently usable by other platform agents |
| Demo quality | End-to-end FHIR patient → clinical output in under 12 seconds |

---

## 12. Submission Checklist

- [ ] Orchestrator registered as A2A agent on Prompt Opinion platform
- [ ] Drug Safety MCP published to Prompt Opinion Marketplace
- [ ] HAPI FHIR sandbox populated with synthetic patient data (Synthea)
- [ ] All 8 services running via Docker Compose
- [ ] Demo video recorded (under 3 minutes)
- [ ] GitHub repository public with README
- [ ] SHARP extension spec correctly implemented
- [ ] All FHIR output resources validated against R4 schema
- [ ] Medical disclaimer included in all AI-generated clinical outputs

---

## 13. Honest Risk Assessment

| Risk | Mitigation |
|---|---|
| FHIR data quality from sandbox | Use Synthea for clean synthetic data; handle missing fields gracefully |
| CNN model accuracy on real X-rays | Frame as "AI triage assistance, not radiologist replacement" |
| Digital Twin model trained on 100 patients | Be transparent; frame as architecture demo with real production path |
| 12-second latency | Use Redis caching; run Diagnosis + Lab in parallel; pre-warm models |
| A2A platform integration complexity | Start integration in week 1; use Prompt Opinion overview video |