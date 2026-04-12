"""
MediTwin LangGraph Graph — with Streaming Support
Defines the full StateGraph connecting all 8 specialist agents.

Streaming modes:
  - "messages" : LLM token-level streaming (automatic from LangChain LLMs)
  - "updates"  : node-level state updates
  - "custom"   : status events from get_stream_writer() inside each node

Each node emits custom stream events so the client knows exactly what's
happening in real time:
  {"type": "status",   "node": "patient_context",  "message": "Fetching FHIR data..."}
  {"type": "result",   "node": "diagnosis",        "summary": "CAP J18.9 (HIGH)"}
  {"type": "error",    "node": "drug_safety",       "message": "Agent timed out"}
  {"type": "complete", "node": "explanation",       "message": "Analysis complete"}
"""
import asyncio
import logging
import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.config import get_stream_writer

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

from dotenv import load_dotenv
load_dotenv()


# ── Fallback consensus ─────────────────────────────────────────────────────────
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


# ── Node 1: Patient Context ────────────────────────────────────────────────────
async def node_patient_context(state: MediTwinState) -> dict:
    writer = get_stream_writer()
    patient_id = state["patient_id"]

    writer({
        "type": "status",
        "node": "patient_context",
        "message": f"Fetching FHIR data for patient {patient_id}...",
        "step": 1,
        "total_steps": 7,
    })

    logger.info(f"[N1] PatientContext: fetching patient {patient_id}")

    patient_state = await call_patient_context(
        patient_id=patient_id,
        fhir_base_url=state["fhir_base_url"],
        sharp_token=state.get("sharp_token", ""),
    )

    if patient_state is None:
        writer({
            "type": "error",
            "node": "patient_context",
            "message": "Failed to fetch patient data from FHIR server.",
        })
        logger.error("[N1] PatientContext FAILED — cannot proceed")
        return {
            "patient_state": None,
            "error_log": ["FATAL: Patient Context Agent failed — graph cannot proceed"],
        }

    conditions_count = len(patient_state.get("active_conditions", []))
    labs_count = len(patient_state.get("lab_results", []))
    meds_count = len(patient_state.get("medications", []))

    writer({
        "type": "result",
        "node": "patient_context",
        "message": f"Patient data loaded — {conditions_count} conditions, {labs_count} labs, {meds_count} medications",
        "summary": {
            "patient_name": patient_state.get("demographics", {}).get("name", "Unknown"),
            "age": patient_state.get("demographics", {}).get("age"),
            "conditions": conditions_count,
            "labs": labs_count,
            "medications": meds_count,
            "allergies": len(patient_state.get("allergies", [])),
        },
    })

    logger.info(f"[N1] PatientContext: success — {labs_count} labs, {conditions_count} conditions")
    return {"patient_state": patient_state, "error_log": []}


# ── Node 2: Parallel Diagnosis + Lab ──────────────────────────────────────────
async def node_parallel_diagnosis_lab(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    if state.get("patient_state") is None:
        writer({
            "type": "error",
            "node": "parallel_diagnosis_lab",
            "message": "Skipping diagnosis and lab — no patient state available.",
        })
        return {"diagnosis_output": None, "lab_output": None, "error_log": []}

    writer({
        "type": "status",
        "node": "parallel_diagnosis_lab",
        "message": "Running Diagnosis Agent + Lab Analysis in parallel...",
        "step": 2,
        "total_steps": 7,
    })

    patient_state = state["patient_state"]
    chief_complaint = state.get("chief_complaint", "Not specified")

    logger.info("[N2] Starting Diagnosis + Lab Analysis in parallel")

    diagnosis_result, lab_result = await asyncio.gather(
        call_diagnosis(patient_state, chief_complaint),
        call_lab_analysis(patient_state, None),
        return_exceptions=False,
    )

    errors = []

    if diagnosis_result is None:
        errors.append("WARNING: Diagnosis Agent failed")
        writer({
            "type": "error",
            "node": "diagnosis",
            "message": "Diagnosis Agent returned no result.",
        })
        logger.warning("[N2] Diagnosis Agent returned None")
    else:
        top_dx = diagnosis_result.get("top_diagnosis", "Unknown")
        top_code = diagnosis_result.get("top_icd10_code", "?")
        confidence = diagnosis_result.get("confidence_level", "?")
        penicillin_flag = diagnosis_result.get("penicillin_allergy_flagged", False)
        sepsis_flag = diagnosis_result.get("high_suspicion_sepsis", False)

        flags = []
        if penicillin_flag:
            flags.append("Penicillin allergy flagged")
        if sepsis_flag:
            flags.append("High sepsis suspicion")

        writer({
            "type": "result",
            "node": "diagnosis",
            "message": f"Diagnosis complete — {top_dx} ({top_code}) [{confidence}]",
            "summary": {
                "top_diagnosis": top_dx,
                "icd10_code": top_code,
                "confidence": confidence,
                "differential_count": len(diagnosis_result.get("differential_diagnosis", [])),
                "flags": flags,
                "next_steps_count": len(diagnosis_result.get("recommended_next_steps", [])),
            },
        })
        logger.info(f"[N2] Diagnosis: {top_dx} ({confidence})")

    if lab_result is None:
        errors.append("WARNING: Lab Analysis Agent failed")
        writer({
            "type": "error",
            "node": "lab_analysis",
            "message": "Lab Analysis Agent returned no result.",
        })
        logger.warning("[N2] Lab Analysis Agent returned None")
    else:
        lab_summary = lab_result.get("lab_summary", {})
        severity = lab_summary.get("overall_severity", "UNKNOWN")
        abnormal = lab_summary.get("abnormal_count", 0)
        critical = lab_summary.get("critical_count", 0)
        alerts = lab_result.get("critical_alerts", [])

        alert_msgs = [a.get("message", "")[:60] for a in alerts[:2]]

        writer({
            "type": "result",
            "node": "lab_analysis",
            "message": f"Lab analysis complete — {severity} severity ({abnormal} abnormal, {critical} critical)",
            "summary": {
                "severity": severity,
                "total_results": lab_summary.get("total_results", 0),
                "abnormal_count": abnormal,
                "critical_count": critical,
                "critical_alerts": alert_msgs,
                "confirms_diagnosis": lab_result.get("diagnosis_confirmation", {}).get("confirms_top_diagnosis"),
            },
        })
        logger.info(f"[N2] Lab: {severity} severity, {critical} critical")

    return {
        "diagnosis_output": diagnosis_result,
        "lab_output": lab_result,
        "error_log": errors,
    }


# ── Conditional Edge: Imaging Check ───────────────────────────────────────────
def check_imaging_available(state: MediTwinState) -> Literal["imaging_triage", "drug_safety"]:
    if state.get("imaging_available") and state.get("image_data"):
        logger.info("[Router] Image available → routing to imaging_triage")
        return "imaging_triage"
    logger.info("[Router] No image → skipping imaging_triage")
    return "drug_safety"


# ── Node 3: Imaging Triage ─────────────────────────────────────────────────────
async def node_imaging_triage(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "status",
        "node": "imaging_triage",
        "message": "Running CNN chest X-ray analysis (EfficientNetB0)...",
        "step": 3,
        "total_steps": 7,
    })

    patient_state = state.get("patient_state")
    image_data = state.get("image_data")

    if not patient_state or not image_data:
        writer({
            "type": "error",
            "node": "imaging_triage",
            "message": "Imaging skipped — missing patient state or image data.",
        })
        return {"imaging_output": None, "error_log": []}

    logger.info("[N3] ImagingTriage: running CNN inference")
    imaging_result = await call_imaging(patient_state, image_data)

    if imaging_result is None:
        writer({
            "type": "error",
            "node": "imaging_triage",
            "message": "Imaging Triage Agent returned no result.",
        })
        logger.warning("[N3] Imaging Agent returned None")
        return {"imaging_output": None, "error_log": ["WARNING: Imaging Triage Agent failed"]}

    pred = imaging_result.get("model_output", {}).get("prediction", "?")
    conf = imaging_result.get("model_output", {}).get("confidence", 0)
    triage = imaging_result.get("severity_assessment", {}).get("triage_label", "?")
    priority = imaging_result.get("severity_assessment", {}).get("triage_priority", 4)
    findings = imaging_result.get("imaging_findings", {}).get("pattern", "N/A")

    writer({
        "type": "result",
        "node": "imaging_triage",
        "message": f"X-ray analysis complete — {pred} ({conf:.0%} confidence) | {triage}",
        "summary": {
            "prediction": pred,
            "confidence": round(conf, 3),
            "triage_label": triage,
            "triage_priority": priority,
            "findings": findings,
            "confirms_diagnosis": imaging_result.get("confirms_diagnosis"),
            "mock": imaging_result.get("mock", False),
        },
    })

    logger.info(f"[N3] ImagingTriage: {pred} ({conf:.0%} confidence)")
    return {"imaging_output": imaging_result, "error_log": []}


# ── Node 4: Drug Safety ────────────────────────────────────────────────────────
async def node_drug_safety(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "status",
        "node": "drug_safety",
        "message": "Checking drug safety — allergies, interactions, FDA warnings...",
        "step": 4,
        "total_steps": 7,
    })

    patient_state = state.get("patient_state")
    if not patient_state:
        writer({
            "type": "error",
            "node": "drug_safety",
            "message": "Drug safety skipped — no patient state.",
        })
        return {"drug_safety_output": None, "error_log": []}

    logger.info("[N4] DrugSafety: checking medication safety")
    drug_result = await call_drug_safety(patient_state, state.get("diagnosis_output"))

    if drug_result is None:
        writer({
            "type": "error",
            "node": "drug_safety",
            "message": "Drug Safety Agent returned no result.",
        })
        logger.warning("[N4] Drug Safety Agent returned None")
        return {"drug_safety_output": None, "error_log": ["WARNING: Drug Safety Agent failed"]}

    status = drug_result.get("safety_status", "?")
    approved = drug_result.get("approved_medications", [])
    flagged = drug_result.get("flagged_medications", [])
    interactions = drug_result.get("critical_interactions", [])
    contras = drug_result.get("contraindications", [])

    

    writer({
        "type": "result",
        "node": "drug_safety",
        "message": f"Drug safety: {status} — {len(approved)} approved, {len(flagged)} flagged",
        "summary": {
            "safety_status": status,
            "approved_medications": approved,
            "flagged_medications": flagged,
            "interaction_count": len(interactions),
            "contraindication_count": len(contras),
            "fda_black_box_count": drug_result.get("summary", {}).get("black_box_warnings", 0),
        },
    })

    logger.info(f"[N4] DrugSafety: {status}, {len(flagged)} medications flagged")
    return {"drug_safety_output": drug_result, "error_log": []}


# ── Node 5: Digital Twin ───────────────────────────────────────────────────────
async def node_digital_twin(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "status",
        "node": "digital_twin",
        "message": "Running Digital Twin outcome simulation (XGBoost)...",
        "step": 5,
        "total_steps": 7,
    })

    patient_state = state.get("patient_state")
    if not patient_state:
        writer({
            "type": "error",
            "node": "digital_twin",
            "message": "Digital Twin skipped — no patient state.",
        })
        return {"digital_twin_output": None, "error_log": []}

    logger.info("[N5] DigitalTwin: running outcome simulation")
    twin_result = await call_digital_twin(
        patient_state,
        state.get("diagnosis_output"),
        state.get("drug_safety_output"),
    )

    if twin_result is None:
        writer({
            "type": "error",
            "node": "digital_twin",
            "message": "Digital Twin Agent returned no result.",
        })
        logger.warning("[N5] Digital Twin Agent returned None")
        return {"digital_twin_output": None, "error_log": ["WARNING: Digital Twin Agent failed"]}

    sim_summary = twin_result.get("simulation_summary", {})
    risk = sim_summary.get("patient_risk_profile", "?")
    rec = sim_summary.get("recommended_option", "?")
    rec_conf = sim_summary.get("recommendation_confidence", 0)
    baseline = sim_summary.get("baseline_risks", {})
    scenarios = twin_result.get("scenarios", [])

    # Find recommended scenario details
    rec_scenario = next((s for s in scenarios if s.get("option_id") == rec), None)
    rec_recovery = rec_scenario.get("predictions", {}).get("recovery_probability_7d", 0) if rec_scenario else 0

    writer({
        "type": "result",
        "node": "digital_twin",
        "message": f"Simulation complete — {risk} risk | Recommended: Option {rec} ({rec_conf:.0%} confidence)",
        "summary": {
            "risk_profile": risk,
            "recommended_option": rec,
            "recommendation_confidence": round(rec_conf, 3),
            "baseline_mortality_30d": round(baseline.get("mortality_30d", 0), 3),
            "baseline_readmission_30d": round(baseline.get("readmission_30d", 0), 3),
            "predicted_7d_recovery": round(rec_recovery, 3),
            "scenarios_count": len(scenarios),
            "model_confidence": twin_result.get("model_confidence", "?"),
        },
    })

    logger.info(f"[N5] DigitalTwin: risk={risk}, recommended=Option {rec}")
    return {"digital_twin_output": twin_result, "error_log": []}


# ── Node 6: Consensus ──────────────────────────────────────────────────────────
async def node_consensus(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "status",
        "node": "consensus",
        "message": "Running conflict detection and consensus arbitration...",
        "step": 6,
        "total_steps": 7,
    })

    patient_state = state.get("patient_state")
    if not patient_state:
        fallback = _fallback_consensus(state)
        writer({
            "type": "result",
            "node": "consensus",
            "message": "Using fallback consensus — no patient state.",
            "summary": {"status": fallback["consensus_status"]},
        })
        return {
            "consensus_output": fallback,
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
        writer({
            "type": "error",
            "node": "consensus",
            "message": "Consensus Agent unavailable — using fallback.",
            "summary": {"status": "FALLBACK"},
        })
        return {
            "consensus_output": fallback,
            "human_review_required": False,
            "error_log": ["WARNING: Consensus Agent failed — using diagnosis output directly"],
        }

    status = consensus_result.get("consensus_status", "?")
    conf = consensus_result.get("aggregate_confidence", 0)
    conflicts = consensus_result.get("conflict_count", 0)
    human = consensus_result.get("human_review_required", False)
    final_dx = consensus_result.get("final_diagnosis", "?")

    writer({
        "type": "result",
        "node": "consensus",
        "message": f"Consensus: {status} | Confidence: {conf:.0%} | Conflicts: {conflicts}",
        "summary": {
            "status": status,
            "aggregate_confidence": round(conf, 3),
            "conflict_count": conflicts,
            "human_review_required": human,
            "final_diagnosis": final_dx,
            "conflicts": consensus_result.get("conflicts", []),
        },
    })

    logger.info(f"[N6] Consensus: {status}, confidence={conf:.0%}, human_review={human}")
    return {
        "consensus_output": consensus_result,
        "human_review_required": consensus_result.get("human_review_required", False),
        "error_log": [],
    }


# ── Node 7: Explanation ────────────────────────────────────────────────────────
async def node_explanation(state: MediTwinState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "status",
        "node": "explanation",
        "message": "Generating SOAP note, patient explanation, and FHIR bundle...",
        "step": 7,
        "total_steps": 7,
    })

    patient_state = state.get("patient_state")
    consensus_output = state.get("consensus_output")

    if not patient_state or not consensus_output:
        writer({
            "type": "error",
            "node": "explanation",
            "message": "Explanation skipped — missing required inputs.",
        })
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
        writer({
            "type": "error",
            "node": "explanation",
            "message": "Explanation Agent returned no result.",
        })
        logger.error("[N7] Explanation Agent returned None")
        return {
            "final_output": {
                "error": "Explanation Agent failed",
                "consensus_output": consensus_output,
                "partial_outputs_available": True,
            },
            "error_log": ["ERROR: Explanation Agent failed"],
        }

    # Extract key outputs for the stream summary
    soap = explanation_result.get("clinician_output", {}).get("soap_note", {})
    plan_count = len(soap.get("plan", []))
    fhir_count = explanation_result.get("fhir_bundle", {}).get("_entry_count", 0)
    reading_level = explanation_result.get("reading_level_check", {}).get("grade_level", "?")
    risk_factors = len(explanation_result.get("risk_attribution", {}).get("shap_style_breakdown", []))

    writer({
        "type": "complete",
        "node": "explanation",
        "message": "MediTwin analysis complete! SOAP note, patient explanation, and FHIR bundle ready.",
        "summary": {
            "soap_plan_items": plan_count,
            "fhir_resources": fhir_count,
            "patient_explanation_reading_level": reading_level,
            "risk_attribution_factors": risk_factors,
            "consensus_status": consensus_output.get("consensus_status"),
            "human_review_required": state.get("human_review_required", False),
        },
    })

    logger.info("[N7] Explanation complete — MediTwin analysis done")
    return {"final_output": explanation_result, "error_log": []}


async def build_meditwin_graph_with_checkpointer(checkpointer) -> StateGraph:
    """
    Accepts an already-initialized checkpointer.
    Connection lifecycle is managed by the caller (main.py lifespan).
    """
    try:
        await checkpointer.setup()
        logger.info("✓ PostgreSQL checkpointer initialized")
    except Exception as e:
        logger.warning(f"Checkpointer setup: {e} (tables may already exist)")

    graph = StateGraph(MediTwinState)

    graph.add_node("patient_context",        node_patient_context)
    graph.add_node("parallel_diagnosis_lab", node_parallel_diagnosis_lab)
    graph.add_node("imaging_triage",         node_imaging_triage)
    graph.add_node("drug_safety",            node_drug_safety)
    graph.add_node("digital_twin",           node_digital_twin)
    graph.add_node("consensus",              node_consensus)
    graph.add_node("explanation",            node_explanation)

    graph.add_edge(START, "patient_context")
    graph.add_edge("patient_context", "parallel_diagnosis_lab")
    graph.add_conditional_edges(
        "parallel_diagnosis_lab",
        check_imaging_available,
        {"imaging_triage": "imaging_triage", "drug_safety": "drug_safety"},
    )
    graph.add_edge("imaging_triage", "drug_safety")
    graph.add_edge("drug_safety",   "digital_twin")
    graph.add_edge("digital_twin",  "consensus")
    graph.add_edge("consensus",     "explanation")
    graph.add_edge("explanation",   END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("✓ MediTwin graph compiled with PostgreSQL checkpointer")
    return compiled


# ── Legacy sync builder (for tests) ───────────────────────────────────────────
def build_meditwin_graph() -> StateGraph:
    logger.warning("Using legacy build_meditwin_graph() without checkpointer")

    graph = StateGraph(MediTwinState)

    graph.add_node("patient_context",          node_patient_context)
    graph.add_node("parallel_diagnosis_lab",   node_parallel_diagnosis_lab)
    graph.add_node("imaging_triage",           node_imaging_triage)
    graph.add_node("drug_safety",              node_drug_safety)
    graph.add_node("digital_twin",             node_digital_twin)
    graph.add_node("consensus",                node_consensus)
    graph.add_node("explanation",              node_explanation)

    graph.add_edge(START, "patient_context")
    graph.add_edge("patient_context", "parallel_diagnosis_lab")
    graph.add_conditional_edges(
        "parallel_diagnosis_lab",
        check_imaging_available,
        {"imaging_triage": "imaging_triage", "drug_safety": "drug_safety"},
    )
    graph.add_edge("imaging_triage", "drug_safety")
    graph.add_edge("drug_safety",   "digital_twin")
    graph.add_edge("digital_twin",  "consensus")
    graph.add_edge("consensus",     "explanation")
    graph.add_edge("explanation",   END)

    return graph.compile()