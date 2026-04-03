# Agent 9: Orchestrator Agent

**Role:** System Brain — A2A Coordination, Sequencing, and Context Passing  
**Type:** A2A Agent (Primary Entry Point)  
**Framework:** LangGraph StateGraph + FastAPI  
**Protocol:** A2A + SHARP context propagation

---

## What This Agent Does

The Orchestrator is the **central nervous system of MediTwin**. It is the only agent that the Prompt Opinion platform interacts with directly. Everything else runs inside it.

Its job is to:
1. Accept the initial request from the Prompt Opinion platform (patient ID + SHARP token)
2. Sequence all specialist agents in the correct order
3. Pass the right data between agents at each step
4. Route through the Consensus Agent's conflict detection
5. Collect all outputs and hand them to the Explanation Agent
6. Return the final result to the platform

It is implemented as a **LangGraph `StateGraph`** — each node in the graph is one specialist agent, and edges define the flow.

---

## Responsibilities

1. Register as an A2A agent on the Prompt Opinion platform
2. Accept SHARP context (patient ID + FHIR token) from the platform
3. Execute the agent graph in order with parallel where possible
4. Carry state across all nodes using LangGraph's shared state
5. Handle partial failures gracefully (if imaging fails, continue without it)
6. Return the complete MediTwin output as a structured response

---

## A2A Agent Card (Prompt Opinion Registration)

This is the metadata you publish to the Prompt Opinion Marketplace:
```json
{
  "name": "MediTwin AI",
  "version": "1.0.0",
  "description": "Multi-agent clinical decision support system. Combines diagnosis reasoning, lab interpretation, medical imaging analysis, drug safety checking, and digital twin simulation to produce a comprehensive clinical assessment and treatment plan.",
  "author": "Tayyab Hussain",
  "input_schema": {
    "type": "object",
    "properties": {
      "patient_id": { "type": "string", "description": "FHIR Patient resource ID" },
      "chief_complaint": { "type": "string", "description": "Primary reason for visit" },
      "include_imaging": { "type": "boolean", "default": false },
      "image_data": { "type": "string", "description": "Base64 chest X-ray (optional)" }
    },
    "required": ["patient_id", "chief_complaint"]
  },
  "sharp_context": {
    "consumes": ["patient_id", "fhir_token", "fhir_base_url"],
    "required": true
  },
  "capabilities": [
    "differential_diagnosis",
    "lab_interpretation",
    "drug_interaction_checking",
    "medical_imaging_analysis",
    "outcome_simulation",
    "fhir_r4_compliant_output"
  ]
}
```

---

## LangGraph State Definition

```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph import add_messages

class MediTwinState(TypedDict):
    # Input
    patient_id: str
    chief_complaint: str
    sharp_token: str
    fhir_base_url: str
    image_data: Optional[str]
    
    # Agent outputs (populated as graph executes)
    patient_state: Optional[dict]
    diagnosis_output: Optional[dict]
    lab_output: Optional[dict]
    imaging_output: Optional[dict]
    drug_safety_output: Optional[dict]
    treatment_options: Optional[list]
    digital_twin_output: Optional[dict]
    detected_conflicts: Optional[list]
    consensus_output: Optional[dict]
    final_output: Optional[dict]
    
    # Control flags
    imaging_available: bool
    human_review_required: bool
    error_log: list
    
    # LangGraph message history
    messages: Annotated[list, add_messages]
```

---

## LangGraph Graph Definition

```python
from langgraph.graph import StateGraph, END

def build_meditwin_graph() -> StateGraph:
    graph = StateGraph(MediTwinState)
    
    # Add all nodes
    graph.add_node("patient_context", run_patient_context_agent)
    graph.add_node("parallel_diagnosis_lab", run_diagnosis_and_lab_parallel)
    graph.add_node("imaging_check", check_imaging_available)
    graph.add_node("imaging_triage", run_imaging_agent)
    graph.add_node("drug_safety", run_drug_safety_agent)
    graph.add_node("digital_twin", run_digital_twin_agent)
    graph.add_node("consensus", run_consensus_agent)
    graph.add_node("explanation", run_explanation_agent)
    
    # Define edges (the flow)
    graph.set_entry_point("patient_context")
    graph.add_edge("patient_context", "parallel_diagnosis_lab")
    graph.add_edge("parallel_diagnosis_lab", "imaging_check")
    
    # Conditional: run imaging only if X-ray was provided
    graph.add_conditional_edges(
        "imaging_check",
        lambda state: "imaging_triage" if state["imaging_available"] else "drug_safety",
        {"imaging_triage": "imaging_triage", "drug_safety": "drug_safety"}
    )
    
    graph.add_edge("imaging_triage", "drug_safety")
    graph.add_edge("drug_safety", "digital_twin")
    graph.add_edge("digital_twin", "consensus")
    
    # Consensus routing
    graph.add_conditional_edges(
        "consensus",
        route_after_consensus,
        {
            "explanation": "explanation",
            "escalation": "explanation"  # Explanation agent handles escalation output too
        }
    )
    
    graph.add_edge("explanation", END)
    
    return graph.compile()
```

---

## Parallel Execution Node

Diagnosis and Lab run in parallel — no dependency between them:
```python
import asyncio

async def run_diagnosis_and_lab_parallel(state: MediTwinState) -> MediTwinState:
    diagnosis_task = asyncio.create_task(
        call_diagnosis_agent(state["patient_state"], state["chief_complaint"])
    )
    lab_task = asyncio.create_task(
        call_lab_agent(state["patient_state"])
    )
    
    diagnosis_output, lab_output = await asyncio.gather(diagnosis_task, lab_task)
    
    return {
        **state,
        "diagnosis_output": diagnosis_output,
        "lab_output": lab_output
    }
```

---

## Agent Call Pattern

Each node calls its specialist agent via HTTP (they run as separate FastAPI services):
```python
async def call_diagnosis_agent(patient_state: dict, chief_complaint: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{DIAGNOSIS_AGENT_URL}/diagnose",
            json={
                "patient_state": patient_state,
                "chief_complaint": chief_complaint
            },
            headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"}
        )
        return response.json()
```

---

## Error Handling Strategy

```python
async def safe_agent_call(agent_fn, *args, **kwargs) -> dict | None:
    try:
        return await agent_fn(*args, **kwargs)
    except httpx.TimeoutException:
        logger.warning(f"Agent timeout: {agent_fn.__name__}")
        return None
    except Exception as e:
        logger.error(f"Agent error: {agent_fn.__name__} — {e}")
        return None
```

Principle: **No single agent failure should crash the entire system.** If imaging fails, continue. If Digital Twin fails, log and continue. Only Patient Context failure is fatal (no patient data = cannot proceed).

---

## Execution Timeline

```
T+0.0s   Request received, SHARP context extracted
T+0.5s   Patient Context Agent completes (parallel FHIR fetches)
T+0.5s   Diagnosis + Lab agents start (parallel)
T+3.5s   Diagnosis + Lab agents complete
T+3.5s   Imaging agent starts (if X-ray provided)
T+6.0s   Imaging agent completes
T+6.0s   Drug Safety agent starts
T+7.0s   Drug Safety agent completes
T+7.0s   Digital Twin simulation starts
T+9.0s   Digital Twin completes
T+9.0s   Consensus check runs
T+9.5s   Explanation agent starts
T+12.0s  Final output returned

Total: ~12 seconds for full analysis with imaging
       ~8 seconds without imaging
```

---

## FastAPI Entry Point

```python
from fastapi import FastAPI, Header
from langgraph.graph import StateGraph

app = FastAPI(title="MediTwin AI Orchestrator")
meditwin_graph = build_meditwin_graph()

@app.post("/analyze")
async def analyze_patient(
    request: MediTwinRequest,
    x_sharp_patient_id: str = Header(None),
    x_sharp_fhir_token: str = Header(None),
    x_sharp_fhir_base_url: str = Header(None)
):
    initial_state = MediTwinState(
        patient_id=x_sharp_patient_id or request.patient_id,
        chief_complaint=request.chief_complaint,
        sharp_token=x_sharp_fhir_token or "",
        fhir_base_url=x_sharp_fhir_base_url or "https://hapi.fhir.org/baseR4",
        image_data=request.image_data,
        imaging_available=bool(request.image_data),
        human_review_required=False,
        error_log=[]
    )
    
    final_state = await meditwin_graph.ainvoke(initial_state)
    return final_state["final_output"]

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": 8}
```

---

## Service Architecture

Each agent runs as a separate FastAPI service on different ports:

| Service | Port | Docker container |
|---|---|---|
| Orchestrator | 8000 | `meditwin-orchestrator` |
| Patient Context Agent | 8001 | `meditwin-fhir` |
| Diagnosis Agent | 8002 | `meditwin-diagnosis` |
| Lab Analysis Agent | 8003 | `meditwin-lab` |
| Drug Safety MCP | 8004 | `meditwin-drug-safety` |
| Imaging Triage Agent | 8005 | `meditwin-imaging` |
| Digital Twin Agent | 8006 | `meditwin-digital-twin` |
| Explanation Agent | 8007 | `meditwin-explanation` |
| Redis | 6379 | `meditwin-redis` |

Use Docker Compose for local development. Deploy to a VPS or cloud instance for the hackathon demo.

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph framework | LangGraph |
| HTTP calls | httpx (async) |
| API framework | FastAPI |
| State management | LangGraph TypedDict |
| Parallel execution | `asyncio.gather()` |
| Service orchestration | Docker Compose |

---

## Your Existing Skills That Apply

- LangGraph from AINutritionChef (multi-agent pipeline with validation loops)
- FastAPI REST API design
- Docker deployment from your sales management app
- Multi-agent workflow design from 6+ hackathon prototypes