# Agent 7: Consensus + Escalation Agent

**Role:** Disagreement Detection — Quality Control Layer  
**Type:** A2A Agent (LangGraph Conditional Node)  
**Framework:** LangGraph + FastAPI  
**Pattern:** Conditional routing with confidence-weighted arbitration

---

## What This Agent Does

The Consensus Agent is the **quality control brain** of MediTwin. Its job is to monitor all specialist agent outputs and detect when they disagree with each other — then either resolve the disagreement automatically or escalate to human review.

This agent is what separates a multi-agent *system* from a collection of parallel API calls. Without it, conflicting outputs would silently coexist and confuse the clinician. With it, every contradiction is caught, reasoned about, and either resolved or flagged.

It is implemented as a **LangGraph conditional node** — not a standalone microservice — living inside the orchestrator's graph.

---

## Responsibilities

1. Collect outputs from Diagnosis Agent and Lab Analysis Agent
2. Detect diagnosis conflicts (different top conditions)
3. Detect confidence conflicts (same condition, wide confidence gap)
4. Detect treatment conflicts (Drug Safety Agent rejects Treatment Planning Agent's recommendation)
5. Run tiebreaker reasoning if conflict is resolvable
6. Escalate to human review flag if conflict is unresolvable
7. Produce a single consensus clinical assessment

---

## Input

Receives all specialist agent outputs simultaneously from the Orchestrator:
```json
{
  "diagnosis_agent": {
    "top_diagnosis": "J18.9",
    "display": "Community-acquired pneumonia",
    "confidence": 0.87,
    "differential": [...]
  },
  "lab_agent": {
    "confirms_top_diagnosis": true,
    "diagnosis_code": "J18.9",
    "lab_confidence_boost": 0.12,
    "pattern_analysis": {...}
  },
  "imaging_agent": {
    "prediction": "PNEUMONIA",
    "confidence": 0.923,
    "confirms_diagnosis": true,
    "diagnosis_code": "J18.9"
  },
  "drug_safety_agent": {
    "safety_status": "UNSAFE",
    "critical_interactions": [...],
    "contraindications": [...]
  },
  "digital_twin_agent": {
    "recommended_option": "B",
    "recommendation_confidence": 0.81
  }
}
```

---

## Output

### Case A: Full Consensus (no conflict)
```json
{
  "consensus_status": "FULL_CONSENSUS",
  "final_diagnosis": "J18.9 — Community-acquired pneumonia",
  "aggregate_confidence": 0.91,
  "confidence_components": {
    "diagnosis_agent": 0.87,
    "lab_confirmation": 0.12,
    "imaging_confirmation": 0.923
  },
  "human_review_required": false,
  "escalation_flag": null,
  "consensus_summary": "All diagnostic modalities agree: community-acquired pneumonia with high confidence (91%). Lab and imaging findings independently confirm the diagnosis."
}
```

### Case B: Conflict Detected and Resolved
```json
{
  "consensus_status": "CONFLICT_RESOLVED",
  "conflict": {
    "type": "diagnosis_lab_mismatch",
    "description": "Diagnosis agent suggests viral pneumonia (J12.9), but lab shows bacterial markers (high WBC, elevated CRP)",
    "conflict_severity": "MODERATE"
  },
  "resolution": {
    "method": "TIEBREAKER_RAG",
    "resolved_diagnosis": "J18.9 — Bacterial pneumonia",
    "resolution_reasoning": "Lab pattern (WBC 18.4, CRP elevated, neutrophilia 82%) is more specific for bacterial etiology than viral. Literature supports bacterial diagnosis given this lab constellation.",
    "confidence_after_resolution": 0.79
  },
  "human_review_required": false,
  "consensus_summary": "Conflict resolved in favor of bacterial etiology based on lab evidence."
}
```

### Case C: Unresolvable Conflict — Escalation
```json
{
  "consensus_status": "ESCALATION_REQUIRED",
  "conflict": {
    "type": "irreconcilable_evidence",
    "description": "Imaging shows no consolidation (NORMAL, 78% confidence) but clinical presentation and labs strongly suggest pneumonia",
    "conflict_severity": "HIGH"
  },
  "escalation_flag": {
    "priority": "URGENT",
    "reason": "Imaging-clinical dissociation detected. Possible early pneumonia not yet visible on X-ray, or alternative diagnosis (PE, lung mass) must be excluded.",
    "recommended_actions": [
      "Radiologist review of chest X-ray",
      "Consider CT chest if clinical suspicion remains high",
      "Consider pulmonary embolism workup (D-dimer)"
    ]
  },
  "human_review_required": true,
  "partial_outputs_available": true
}
```

---

## How It Works — Step by Step

### Step 1: Conflict Detection Rules
```python
def detect_conflicts(agent_outputs: dict) -> list[Conflict]:
    conflicts = []
    
    diag = agent_outputs["diagnosis_agent"]
    lab = agent_outputs["lab_agent"]
    imaging = agent_outputs.get("imaging_agent")
    
    # Rule 1: Diagnosis-Lab disagreement
    if lab["confirms_top_diagnosis"] == False:
        lab_suggested = lab.get("alternative_diagnosis_code")
        if lab_suggested and lab_suggested != diag["top_diagnosis"]:
            conflicts.append(Conflict(
                type="diagnosis_lab_mismatch",
                agent_a="diagnosis", output_a=diag["top_diagnosis"],
                agent_b="lab", output_b=lab_suggested,
                severity=compute_severity(diag["confidence"], lab["lab_confidence_boost"])
            ))
    
    # Rule 2: Imaging-Clinical dissociation
    if imaging and imaging["prediction"] == "NORMAL" and diag["confidence"] > 0.75:
        conflicts.append(Conflict(
            type="imaging_clinical_dissociation",
            severity="HIGH",
            description="High clinical suspicion but normal imaging"
        ))
    
    # Rule 3: Drug safety rejection
    drug_safety = agent_outputs.get("drug_safety_agent", {})
    if drug_safety.get("safety_status") == "UNSAFE" and drug_safety.get("contraindications"):
        # Not a diagnosis conflict, but blocks treatment plan
        conflicts.append(Conflict(
            type="treatment_contraindicated",
            severity="MODERATE",
            description="Proposed treatment has critical contraindications"
        ))
    
    return conflicts
```

### Step 2: LangGraph Conditional Routing
```python
from langgraph.graph import StateGraph, END

def route_consensus(state: AgentState) -> str:
    conflicts = state["detected_conflicts"]
    
    if not conflicts:
        return "no_conflict"
    
    max_severity = max(c.severity for c in conflicts)
    
    if max_severity == "HIGH":
        return "escalate"
    else:
        return "resolve"

# Build the LangGraph conditional node
graph.add_conditional_edges(
    "consensus_detector",
    route_consensus,
    {
        "no_conflict": "explanation_agent",
        "resolve": "tiebreaker_node",
        "escalate": "escalation_handler"
    }
)
```

### Step 3: Tiebreaker RAG Query
When conflict is resolvable, run a targeted RAG query:
```python
async def run_tiebreaker(conflict: Conflict, patient_state: dict) -> Resolution:
    tiebreaker_query = f"""
    Clinical conflict: {conflict.description}
    Patient: {patient_summary}
    
    Agent A says: {conflict.output_a} (confidence: {conflict.confidence_a})
    Agent B says: {conflict.output_b} (confidence: {conflict.confidence_b})
    
    Based on clinical guidelines, which interpretation is correct and why?
    """
    
    # Use the same medical knowledge RAG from Diagnosis Agent
    docs = retriever.get_relevant_documents(tiebreaker_query)
    resolution = llm.invoke(f"Context: {docs}\n\nQuestion: {tiebreaker_query}")
    
    return Resolution(method="TIEBREAKER_RAG", reasoning=resolution, ...)
```

### Step 4: Aggregate Confidence Scoring
When consensus is reached, compute weighted aggregate confidence:
```python
def compute_aggregate_confidence(agent_outputs: dict, resolution: Resolution) -> float:
    weights = {
        "diagnosis_agent": 0.35,
        "lab_agent": 0.30,
        "imaging_agent": 0.25,
        "tiebreaker_boost": 0.10
    }
    
    score = 0.0
    score += agent_outputs["diagnosis_agent"]["confidence"] * weights["diagnosis_agent"]
    score += agent_outputs["lab_agent"]["lab_confidence_boost"] * weights["lab_agent"]
    
    if "imaging_agent" in agent_outputs:
        score += agent_outputs["imaging_agent"]["confidence"] * weights["imaging_agent"]
    
    if resolution:
        score += 0.05  # Small boost for resolved conflict (shows system caught and fixed it)
    
    return round(min(score, 0.99), 2)  # Cap at 0.99 — never 100% certain in medicine
```

---

## LangGraph State Definition

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class MediTwinState(TypedDict):
    patient_state: dict
    diagnosis_output: dict
    lab_output: dict
    imaging_output: dict | None
    drug_safety_output: dict
    digital_twin_output: dict
    detected_conflicts: list
    consensus_status: str
    final_assessment: dict
    human_review_required: bool
    messages: Annotated[list, add_messages]
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph framework | LangGraph |
| Conflict detection | Python rules engine |
| Tiebreaker | LangChain RAG (same vector store as Diagnosis Agent) |
| State management | LangGraph TypedDict state |
| API integration | Part of Orchestrator — not standalone service |

---

## Your Existing Skills That Apply

- LangGraph multi-agent pipelines from AINutritionChef (macro validation → auto-adjustment loop)
- Conditional branching in LangGraph
- Prompt engineering for structured reasoning
- The disagreement concept maps directly to your existing multi-step validation pipelines

---

## Design Principle

Keep this agent simple and conservative. When in doubt, escalate. A false escalation (calling for human review when none was needed) is far safer than a missed conflict in a clinical setting. The threshold for escalation should be low.