"""
MediTwin LangGraph Graph
Defines the full StateGraph connecting all 8 specialist agents.

Graph topology (from architecture doc):
  patient_context
      ↓
  parallel_diagnosis_lab  (Diagnosis + Lab run concurrently via asyncio.gather)
      ↓
  [conditional] imaging_check → imaging_triage (only if imaging_available)
      ↓
  drug_safety
      ↓
  digital_twin
      ↓
  consensus
      ↓
  explanation → END

Key design decisions:
  - Parallel execution: Diagnosis + Lab run inside ONE node using asyncio.gather()
    (LangGraph parallel fan-out is for map-reduce; for our 2-parallel case
     asyncio.gather inside a node is simpler and avoids state merge complexity)
  - Conditional imaging: add_conditional_edges routes around imaging if no image
  - safe_call on every agent: no single failure crashes the graph
  - Patient Context failure is the ONLY fatal failure
"""
import asyncio
import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from state import MediTwinState
from agent_callers import (
    call_patient_context,
    call_diagnosis,
    call_lab_analysis,
    call_drug_safety,
    call_imaging,
    call_digital_twin,
    call_consensus,
    call_explanation,
)

logger = logging.getLogger("meditwin.graph")


# ── Fallback consensus for when Consensus Agent is down ───────────────────────
def _fallback_consensus(state: MediTwinState) -> dict:
    return {
        "consensus_status": "FULL_CONSENSUS",
        "final_diagnosis": state.get("diagnosis_output", {}).get("top_diagnosis") if state.get("diagnosis_output") else "Unknown",
        "final_icd10_code": state.get("diagnosis_output", {}).get("top_icd10_code") if state.get("diagnosis_output") else None,
        "aggregate_confidence": 0.60,
        "human_review_required": False,
        "conflict_count": 0,
        "conflicts": [],
        "consensus_summary": "Consensus agent unavailable — using diagnosis agent output directly.",
        "partial_outputs_available": True,
    }


# ── Node functions ─────────────────────────────────────────────────────────────

async def node_patient_context(state: MediTwinState) -> dict:
    """
    Node 1: Fetch patient FHIR data.
    FATAL if this fails — no patient data = cannot proceed.
    """
    logger.info(f"[N1] PatientContext: fetching patient {state['patient_id']}")

    patient_state = await call_patient_context(
        patient_id=state["patient_id"],
        fhir_base_url=state["fhir_base_url"],
        sharp_token=state.get("sharp_token", ""),
    )

    if patient_state is None:
        # Fatal — return error state that triggers graceful shutdown
        logger.error("[N1] PatientContext FAILED — cannot proceed")
        return {
            "patient_state": None,
            "error_log": ["FATAL: Patient Context Agent failed — graph cannot proceed"],
        }

    logger.info(f"[N1] PatientContext: success — "
                f"{len(patient_state.get('lab_results', []))} labs, "
                f"{len(patient_state.get('active_conditions', []))} conditions")

    return {"patient_state": patient_state, "error_log": []}


async def node_parallel_diagnosis_lab(state: MediTwinState) -> dict:
    """
    Node 2: Run Diagnosis and Lab Analysis in parallel.
    Both receive PatientState. Neither depends on the other.
    Uses asyncio.gather() for concurrent HTTP calls.
    """
    if state.get("patient_state") is None:
        logger.warning("[N2] Skipping — no patient state available")
        return {"diagnosis_output": None, "lab_output": None, "error_log": []}

    patient_state = state["patient_state"]
    chief_complaint = state.get("chief_complaint", "Not specified")

    logger.info("[N2] Starting Diagnosis + Lab Analysis in parallel")

    diagnosis_result, lab_result = await asyncio.gather(
        call_diagnosis(patient_state, chief_complaint),
        call_lab_analysis(patient_state, None),  # lab runs independently first
        return_exceptions=False,
    )

    errors = []
    if diagnosis_result is None:
        errors.append("WARNING: Diagnosis Agent failed")
        logger.warning("[N2] Diagnosis Agent returned None")
    else:
        logger.info(f"[N2] Diagnosis: {diagnosis_result.get('top_diagnosis', '?')} "
                    f"({diagnosis_result.get('confidence_level', '?')})")

    if lab_result is None:
        errors.append("WARNING: Lab Analysis Agent failed")
        logger.warning("[N2] Lab Analysis Agent returned None")
    else:
        severity = lab_result.get("lab_summary", {}).get("overall_severity", "?")
        logger.info(f"[N2] Lab: {severity} severity, "
                    f"{lab_result.get('lab_summary', {}).get('critical_count', 0)} critical")

    return {
        "diagnosis_output": diagnosis_result,
        "lab_output": lab_result,
        "error_log": errors,
    }


def check_imaging_available(state: MediTwinState) -> Literal["imaging_triage", "drug_safety"]:
    """
    Conditional edge: route to imaging if image was provided, else skip.
    """
    if state.get("imaging_available") and state.get("image_data"):
        logger.info("[Router] Image available → routing to imaging_triage")
        return "imaging_triage"
    logger.info("[Router] No image → skipping imaging_triage")
    return "drug_safety"


async def node_imaging_triage(state: MediTwinState) -> dict:
    """
    Node 3 (conditional): Run CNN inference on chest X-ray.
    Only reached if imaging_available = True.
    """
    patient_state = state.get("patient_state")
    image_data = state.get("image_data")

    if not patient_state or not image_data:
        return {"imaging_output": None, "error_log": []}

    logger.info("[N3] ImagingTriage: running CNN inference")
    imaging_result = await call_imaging(patient_state, image_data)

    if imaging_result is None:
        logger.warning("[N3] Imaging Agent returned None")
        return {"imaging_output": None, "error_log": ["WARNING: Imaging Triage Agent failed"]}

    pred = imaging_result.get("model_output", {}).get("prediction", "?")
    conf = imaging_result.get("model_output", {}).get("confidence", 0)
    logger.info(f"[N3] ImagingTriage: {pred} ({conf:.0%} confidence)")

    return {"imaging_output": imaging_result, "error_log": []}


async def node_drug_safety(state: MediTwinState) -> dict:
    """
    Node 4: Drug safety check.
    Needs patient state + diagnosis output (for proposed medications).
    """
    patient_state = state.get("patient_state")
    if not patient_state:
        return {"drug_safety_output": None, "error_log": []}

    logger.info("[N4] DrugSafety: checking medication safety")
    drug_result = await call_drug_safety(
        patient_state,
        state.get("diagnosis_output"),
    )

    if drug_result is None:
        logger.warning("[N4] Drug Safety Agent returned None")
        return {"drug_safety_output": None, "error_log": ["WARNING: Drug Safety Agent failed"]}

    status = drug_result.get("safety_status", "?")
    flagged = len(drug_result.get("flagged_medications", []))
    logger.info(f"[N4] DrugSafety: {status}, {flagged} medications flagged")

    return {"drug_safety_output": drug_result, "error_log": []}


async def node_digital_twin(state: MediTwinState) -> dict:
    """
    Node 5: Risk simulation and treatment scenario comparison.
    Needs all previous outputs.
    """
    patient_state = state.get("patient_state")
    if not patient_state:
        return {"digital_twin_output": None, "error_log": []}

    logger.info("[N5] DigitalTwin: running outcome simulation")
    twin_result = await call_digital_twin(
        patient_state,
        state.get("diagnosis_output"),
        state.get("drug_safety_output"),
    )

    if twin_result is None:
        logger.warning("[N5] Digital Twin Agent returned None")
        return {"digital_twin_output": None, "error_log": ["WARNING: Digital Twin Agent failed"]}

    risk = twin_result.get("simulation_summary", {}).get("patient_risk_profile", "?")
    rec = twin_result.get("simulation_summary", {}).get("recommended_option", "?")
    logger.info(f"[N5] DigitalTwin: risk={risk}, recommended=Option {rec}")

    return {"digital_twin_output": twin_result, "error_log": []}


async def node_consensus(state: MediTwinState) -> dict:
    """
    Node 6: Conflict detection and consensus arbitration.
    Reviews all agent outputs. Produces final clinical assessment.
    """
    patient_state = state.get("patient_state")
    if not patient_state:
        return {
            "consensus_output": _fallback_consensus(state),
            "human_review_required": False,
            "error_log": [],
        }

    logger.info("[N6] Consensus: running conflict detection")
    consensus_result = await call_consensus(
        patient_state=patient_state,
        diagnosis_output=state.get("diagnosis_output"),
        lab_output=state.get("lab_output"),
        imaging_output=state.get("imaging_output"),
        drug_safety_output=state.get("drug_safety_output"),
    )

    if consensus_result is None:
        logger.warning("[N6] Consensus Agent returned None — using fallback")
        fallback = _fallback_consensus(state)
        return {
            "consensus_output": fallback,
            "human_review_required": False,
            "error_log": ["WARNING: Consensus Agent failed — using diagnosis output directly"],
        }

    status = consensus_result.get("consensus_status", "?")
    conf = consensus_result.get("aggregate_confidence", 0)
    human = consensus_result.get("human_review_required", False)
    logger.info(f"[N6] Consensus: {status}, confidence={conf:.0%}, human_review={human}")

    return {
        "consensus_output": consensus_result,
        "human_review_required": consensus_result.get("human_review_required", False),
        "error_log": [],
    }


async def node_explanation(state: MediTwinState) -> dict:
    """
    Node 7: Final output generation.
    SOAP note + patient explanation + FHIR Bundle.
    """
    patient_state = state.get("patient_state")
    consensus_output = state.get("consensus_output")

    if not patient_state or not consensus_output:
        logger.error("[N7] Explanation: missing patient_state or consensus_output")
        return {
            "final_output": {
                "error": "Explanation Agent: missing required inputs",
                "partial_outputs": {
                    "diagnosis": state.get("diagnosis_output"),
                    "lab": state.get("lab_output"),
                    "consensus": consensus_output,
                },
            },
            "error_log": ["ERROR: Explanation Agent missing required inputs"],
        }

    logger.info("[N7] Explanation: generating SOAP note + patient explanation + FHIR Bundle")
    explanation_result = await call_explanation(
        patient_state=patient_state,
        consensus_output=consensus_output,
        diagnosis_output=state.get("diagnosis_output"),
        lab_output=state.get("lab_output"),
        imaging_output=state.get("imaging_output"),
        drug_safety_output=state.get("drug_safety_output"),
        digital_twin_output=state.get("digital_twin_output"),
        chief_complaint=state.get("chief_complaint", "Not specified"),
    )

    if explanation_result is None:
        logger.error("[N7] Explanation Agent returned None")
        return {
            "final_output": {
                "error": "Explanation Agent failed",
                "consensus_output": consensus_output,
                "partial_outputs_available": True,
            },
            "error_log": ["ERROR: Explanation Agent failed"],
        }

    logger.info("[N7] Explanation complete — MediTwin analysis done")
    return {"final_output": explanation_result, "error_log": []}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_meditwin_graph() -> StateGraph:
    """
    Build and compile the full MediTwin LangGraph StateGraph.
    Returns the compiled graph, ready for ainvoke().
    """
    graph = StateGraph(MediTwinState)

    # Add all nodes
    graph.add_node("patient_context",          node_patient_context)
    graph.add_node("parallel_diagnosis_lab",   node_parallel_diagnosis_lab)
    graph.add_node("imaging_triage",           node_imaging_triage)
    graph.add_node("drug_safety",              node_drug_safety)
    graph.add_node("digital_twin",             node_digital_twin)
    graph.add_node("consensus",                node_consensus)
    graph.add_node("explanation",              node_explanation)

    # Entry point
    graph.add_edge(START, "patient_context")

    # Sequential: patient_context → parallel_diagnosis_lab
    graph.add_edge("patient_context", "parallel_diagnosis_lab")

    # Conditional: imaging check after diagnosis + lab
    graph.add_conditional_edges(
        "parallel_diagnosis_lab",
        check_imaging_available,
        {
            "imaging_triage": "imaging_triage",
            "drug_safety":    "drug_safety",
        },
    )

    # Imaging → drug_safety (imaging merges back into the main path)
    graph.add_edge("imaging_triage", "drug_safety")

    # Sequential remainder
    graph.add_edge("drug_safety",   "digital_twin")
    graph.add_edge("digital_twin",  "consensus")
    graph.add_edge("consensus",     "explanation")
    graph.add_edge("explanation",   END)

    return graph.compile()