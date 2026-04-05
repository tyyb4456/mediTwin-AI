"""
Agent 7: Consensus + Escalation Agent
Port: 8007

Implementation note from spec + strategy doc:
  "This agent is implemented as a LangGraph conditional node — not a standalone
  microservice — living inside the orchestrator's graph."

However, for hackathon demo purposes we also expose it as a REST endpoint
at /consensus so the Orchestrator can call it over HTTP (same as other agents).
The LangGraph graph definition is in graph.py and is the canonical implementation.
The REST endpoint calls the same core logic.

Pipeline:
  1. Collect all specialist agent outputs
  2. Run conflict detection (pure Python rules — deterministic)
  3. Route: no_conflict → consensus | resolve → tiebreaker RAG | escalate → human flag
  4. Compute weighted aggregate confidence (capped at 0.99)
  5. Return ConsensusOutput for Explanation Agent

Confidence weights (from spec):
  Diagnosis Agent:  0.35
  Lab Agent:        0.30
  Imaging Agent:    0.25
  Tiebreaker boost: 0.05 (small bonus for catching + fixing a conflict)
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from conflict_detector import (
    detect_conflicts,
    route_consensus,
    get_max_severity,
    Conflict,
)
from tiebreaker import tiebreaker


# ── Confidence weights (from spec) ────────────────────────────────────────────
WEIGHTS = {
    "diagnosis":  0.35,
    "lab":        0.30,
    "imaging":    0.25,
    "tiebreaker": 0.05,
}


# ── LangGraph State Definition ─────────────────────────────────────────────────
# Canonical MediTwinState — all fields the graph carries

class MediTwinState(TypedDict):
    # Input
    patient_id: str
    chief_complaint: str

    # Agent outputs (populated as graph executes)
    patient_state: Optional[dict]
    diagnosis_output: Optional[dict]
    lab_output: Optional[dict]
    imaging_output: Optional[dict]
    drug_safety_output: Optional[dict]
    digital_twin_output: Optional[dict]

    # Consensus outputs (populated by this agent)
    detected_conflicts: Optional[list]
    consensus_status: Optional[str]
    final_assessment: Optional[dict]
    human_review_required: bool

    # Control flags
    imaging_available: bool
    error_log: list


# ── Core consensus logic (shared by LangGraph node and REST endpoint) ──────────

def compute_aggregate_confidence(
    diagnosis_output: Optional[dict],
    lab_output: Optional[dict],
    imaging_output: Optional[dict],
    resolution: Optional[dict],
) -> float:
    """
    Weighted aggregate confidence from all agent signals.
    Never returns > 0.99 — we're never 100% certain in medicine.
    """
    score = 0.0

    # Diagnosis Agent confidence (weight 0.35)
    if diagnosis_output:
        dx_conf = 0.0
        differential = diagnosis_output.get("differential_diagnosis", [])
        if differential:
            dx_conf = float(differential[0].get("confidence", 0.0))
        score += dx_conf * WEIGHTS["diagnosis"]

    # Lab Agent confidence boost (weight 0.30)
    if lab_output:
        lab_boost = float(
            lab_output.get("diagnosis_confirmation", {}).get("lab_confidence_boost", 0.0)
        )
        # lab_boost is a delta (e.g. 0.12) — treat it as the lab's confirmation signal
        lab_signal = min(0.85, 0.5 + lab_boost)  # floor at 0.5 if boost is 0
        score += lab_signal * WEIGHTS["lab"]

    # Imaging Agent confidence (weight 0.25) — only if confirms diagnosis
    if imaging_output and not imaging_output.get("mock", False):
        img_confirms = imaging_output.get("confirms_diagnosis", False)
        img_conf = float(imaging_output.get("model_output", {}).get("confidence", 0.0))
        if img_confirms:
            score += img_conf * WEIGHTS["imaging"]
        else:
            # Imaging contradicts — reduce score slightly
            score -= img_conf * WEIGHTS["imaging"] * 0.3

    # Small bonus if system caught and resolved a conflict
    if resolution:
        score += WEIGHTS["tiebreaker"]

    # Cap at 0.99 and floor at 0.30
    return round(min(max(score, 0.30), 0.99), 2)


def build_escalation_actions(conflicts: list[Conflict]) -> list[str]:
    """Build recommended escalation actions from conflict details."""
    actions = set()
    for c in conflicts:
        if c.type == "imaging_clinical_dissociation":
            extra_actions = c.extra.get("recommended_actions", [])
            actions.update(extra_actions)
        elif c.type == "diagnosis_lab_mismatch":
            actions.add("Clinical review of lab results vs clinical presentation")
            actions.add("Consider repeat laboratory testing")
        elif c.type == "treatment_contraindicated":
            if c.extra.get("alternatives_available"):
                actions.add("Review Drug Safety Agent alternatives before prescribing")
            actions.add("Pharmacist consultation for safe medication selection")
    return sorted(actions)


def run_consensus(
    diagnosis_output: Optional[dict],
    lab_output: Optional[dict],
    imaging_output: Optional[dict],
    drug_safety_output: Optional[dict],
    patient_state: Optional[dict],
) -> dict:
    """
    Core consensus pipeline. Shared by LangGraph node and REST endpoint.

    Returns ConsensusOutput dict.
    """
    # Step 1: Detect conflicts
    conflicts = detect_conflicts(
        diagnosis_output=diagnosis_output,
        lab_output=lab_output,
        imaging_output=imaging_output,
        drug_safety_output=drug_safety_output,
    )

    # Step 2: Route
    route = route_consensus(conflicts)

    resolution = None
    consensus_status = "FULL_CONSENSUS"
    human_review_required = False
    escalation_flag = None

    # Step 3: Handle each route
    if route == "no_conflict":
        consensus_status = "FULL_CONSENSUS"

    elif route == "resolve":
        # Attempt tiebreaker RAG for the highest-severity conflict
        primary_conflict = max(
            conflicts,
            key=lambda c: {"HIGH": 2, "MODERATE": 1, "LOW": 0}.get(c.severity, 0)
        )
        resolution = tiebreaker.resolve(primary_conflict, patient_state or {})

        if resolution:
            consensus_status = "CONFLICT_RESOLVED"
        else:
            # Tiebreaker unavailable — fall back to conservative escalation
            consensus_status = "ESCALATION_REQUIRED"
            human_review_required = True

    elif route == "escalate":
        consensus_status = "ESCALATION_REQUIRED"
        human_review_required = True

        escalation_actions = build_escalation_actions(conflicts)
        max_sev = get_max_severity(conflicts)
        escalation_flag = {
            "priority": "URGENT" if max_sev == "HIGH" else "MODERATE",
            "reason": conflicts[0].description if conflicts else "Multiple agent disagreements detected",
            "recommended_actions": escalation_actions,
        }

    # Step 4: Aggregate confidence
    aggregate_confidence = compute_aggregate_confidence(
        diagnosis_output, lab_output, imaging_output, resolution
    )

    # Step 5: Build final assessment
    top_dx = ""
    top_icd10 = ""
    top_dx_display = ""
    if resolution and resolution.get("resolved_diagnosis"):
        top_icd10 = resolution["resolved_diagnosis"]
        top_dx_display = resolution.get("resolved_diagnosis_display", top_icd10)
        top_dx = f"{top_icd10} — {top_dx_display}"
    elif diagnosis_output:
        top_icd10 = diagnosis_output.get("top_icd10_code", "")
        top_dx_display = diagnosis_output.get("top_diagnosis", "")
        top_dx = f"{top_icd10} — {top_dx_display}" if top_icd10 else top_dx_display

    # Build consensus summary
    if consensus_status == "FULL_CONSENSUS":
        sources = []
        if diagnosis_output:
            sources.append("diagnosis")
        if lab_output and lab_output.get("diagnosis_confirmation", {}).get("confirms_top_diagnosis"):
            sources.append("lab analysis")
        if imaging_output and imaging_output.get("confirms_diagnosis") and not imaging_output.get("mock"):
            sources.append("imaging")
        source_str = " and ".join(sources) or "clinical assessment"
        summary = (
            f"All diagnostic modalities agree: {top_dx_display} with "
            f"{aggregate_confidence:.0%} aggregate confidence. "
            f"Consistent findings from {source_str}."
        )
    elif consensus_status == "CONFLICT_RESOLVED":
        summary = (
            f"Conflict detected and resolved: {conflicts[0].description[:100]}. "
            f"Resolution: {resolution.get('reasoning', 'Tiebreaker RAG applied')}."
        )
    else:
        summary = (
            f"Escalation required: {conflicts[0].description[:120] if conflicts else 'Unresolvable conflict'}. "
            "Human clinical review recommended before proceeding."
        )

    return {
        "consensus_status": consensus_status,
        "final_diagnosis": top_dx or None,
        "final_icd10_code": top_icd10 or None,
        "aggregate_confidence": aggregate_confidence,
        "confidence_components": {
            "diagnosis_agent_weight": WEIGHTS["diagnosis"],
            "lab_agent_weight": WEIGHTS["lab"],
            "imaging_agent_weight": WEIGHTS["imaging"],
        },
        "human_review_required": human_review_required,
        "escalation_flag": escalation_flag,
        "conflict_count": len(conflicts),
        "conflicts": [
            {
                "type": c.type,
                "severity": c.severity,
                "description": c.description,
                "agent_a": c.agent_a,
                "output_a": c.output_a,
                "agent_b": c.agent_b,
                "output_b": c.output_b,
            }
            for c in conflicts
        ],
        "resolution": resolution,
        "consensus_summary": summary,
        "partial_outputs_available": True,  # Always true — agents still produced output
    }


# ── LangGraph Node ─────────────────────────────────────────────────────────────
# This is the canonical implementation — a node function for the Orchestrator's StateGraph

def consensus_node(state: MediTwinState) -> dict:
    """
    LangGraph node function.
    Reads all agent outputs from state, runs consensus, writes results back to state.
    """
    result = run_consensus(
        diagnosis_output=state.get("diagnosis_output"),
        lab_output=state.get("lab_output"),
        imaging_output=state.get("imaging_output"),
        drug_safety_output=state.get("drug_safety_output"),
        patient_state=state.get("patient_state"),
    )

    return {
        "detected_conflicts": result["conflicts"],
        "consensus_status": result["consensus_status"],
        "human_review_required": result["human_review_required"],
        "final_assessment": result,
    }


def route_after_consensus(state: MediTwinState) -> str:
    """
    LangGraph conditional edge function.
    Routes to explanation regardless of consensus status
    (Explanation Agent handles all three cases including escalation).
    """
    status = state.get("consensus_status", "FULL_CONSENSUS")
    # Both consensus and escalation go to Explanation Agent
    # Explanation Agent formats the output differently for each case
    if status == "ESCALATION_REQUIRED":
        return "escalation"
    return "explanation"


# ── Minimal demo graph (for testing / documentation) ─────────────────────────

def build_consensus_subgraph() -> StateGraph:
    """
    Minimal LangGraph graph containing just the consensus node.
    In production this is embedded inside the Orchestrator's full graph.
    """
    graph = StateGraph(MediTwinState)
    graph.add_node("consensus", consensus_node)
    graph.add_edge(START, "consensus")
    graph.add_conditional_edges(
        "consensus",
        route_after_consensus,
        {
            "explanation": END,
            "escalation":  END,
        }
    )
    return graph.compile()


# ── FastAPI REST endpoint ──────────────────────────────────────────────────────

class ConsensusRequest(BaseModel):
    diagnosis_output:    Optional[dict] = None
    lab_output:          Optional[dict] = None
    imaging_output:      Optional[dict] = None
    drug_safety_output:  Optional[dict] = None
    patient_state:       Optional[dict] = None


class ConsensusResponse(BaseModel):
    consensus_status:       str
    final_diagnosis:        Optional[str]
    final_icd10_code:       Optional[str]
    aggregate_confidence:   float
    confidence_components:  dict
    human_review_required:  bool
    escalation_flag:        Optional[dict]
    conflict_count:         int
    conflicts:              list
    resolution:             Optional[dict]
    consensus_summary:      str
    partial_outputs_available: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✓ Consensus + Escalation Agent started")
    # Tiebreaker initializes lazily on first use
    yield
    print("✓ Consensus Agent shutdown")


app = FastAPI(
    title="MediTwin Consensus + Escalation Agent",
    description="Conflict detection, tiebreaker RAG, and aggregate confidence scoring",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/consensus", response_model=ConsensusResponse)
async def consensus(request: ConsensusRequest) -> ConsensusResponse:
    """
    Run the full consensus pipeline over all specialist agent outputs.

    Detects conflicts, attempts resolution via RAG if moderate severity,
    escalates if high severity, and returns a weighted aggregate confidence score.
    """
    try:
        result = run_consensus(
            diagnosis_output=request.diagnosis_output,
            lab_output=request.lab_output,
            imaging_output=request.imaging_output,
            drug_safety_output=request.drug_safety_output,
            patient_state=request.patient_state,
        )
        return ConsensusResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consensus failed: {e}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "consensus",
        "version": "1.0.0",
        "conflict_types": [
            "diagnosis_lab_mismatch",
            "imaging_clinical_dissociation",
            "treatment_contraindicated",
        ],
        "routing_options": ["no_conflict", "resolve", "escalate"],
        "tiebreaker_ready": tiebreaker._ready,
        "confidence_weights": WEIGHTS,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8007)