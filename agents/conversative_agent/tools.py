"""
MediTwin Tool Agent — Tool Definitions (DB-backed, patient_id only)
====================================================================
All tools accept only patient_id (str). They query PostgreSQL directly
via db_reader and return the stored result for that patient.

No patient_state dicts, no HTTP round-trips for data retrieval, no
inter-service dependencies for read operations.

DB-backed tools (read stored results):
    fetch_patient_context       → patient_context_results
    run_diagnosis               → diagnosis_results
    analyze_labs                → lab_analysis_results
    check_drug_safety           → drug_safety_results
    analyze_chest_xray          → imaging_triage_results
    simulate_treatment_outcomes → digital_twin_simulations

HTTP tools (live computation — read inputs from DB, POST to service):
    run_consensus               → reads from DB → POST /consensus
    generate_clinical_report    → reads from DB → POST /explain
"""
import os
import logging
import httpx
from typing import Optional
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

import db_reader

logger = logging.getLogger("meditwin.tool_agent.tools")


# ── HTTP service URLs (compute-only tools) ─────────────────────────────────────
CONSENSUS_URL   = os.getenv("CONSENSUS_URL",   "http://localhost:8007")
EXPLANATION_URL = os.getenv("EXPLANATION_URL", "http://localhost:8009")
INTERNAL_TOKEN  = os.getenv("INTERNAL_TOKEN",  "meditwin-internal")
HEADERS = {
    "Authorization": f"Bearer {INTERNAL_TOKEN}",
    "Content-Type": "application/json",
}


async def _post(name: str, url: str, payload: dict, timeout: float = 45.0) -> dict:
    """Async HTTP POST with structured error returns so the LLM can self-correct."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=HEADERS)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        logger.warning(f"[{name}] Timeout after {timeout}s")
        return {
            "error": f"{name} timed out",
            "user_message": f"The {name.lower()} is taking longer than expected.",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"[{name}] HTTP {e.response.status_code}")
        return {
            "error": f"{name} returned HTTP {e.response.status_code}",
            "user_message": f"There was an issue with the {name.lower()} service.",
            "detail": e.response.text[:300],
        }
    except Exception as e:
        logger.error(f"[{name}] Unexpected error: {e}")
        return {
            "error": f"{name} failed",
            "user_message": f"An unexpected error occurred during {name.lower()}.",
            "technical_detail": str(e),
        }


# ── Stream writer helper ───────────────────────────────────────────────────────

def _emit(event: dict):
    """Emit a custom stream event via LangGraph's stream writer."""
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


def _not_found(tool_name: str, patient_id: str, table: str) -> dict:
    msg = (
        f"No stored {table.replace('_', ' ')} found for patient '{patient_id}'. "
        "Run a full clinical analysis through the orchestrator first."
    )
    _emit({"type": "tool_error", "tool": tool_name, "message": f"    ✘   {msg}"})
    return {"error": f"{table}_not_found", "user_message": msg}


# ══════════════════════════════════════════════════════════════════════════════
# DB-BACKED TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def fetch_patient_context(patient_id: str) -> dict:
    """
    Retrieve the complete clinical profile for a patient from stored records.

    Returns demographics, active conditions, medications, allergies, recent
    lab results, diagnostic reports, and imaging availability — everything
    needed to understand the patient's current state.

    This is the starting point for any patient-specific query.

    Args:
        patient_id: The patient's ID (e.g., "example", "patient-123")

    Returns:
        Comprehensive patient record, or an error if no record exists for this patient.
    """
    _emit({
        "type": "tool_start",
        "tool": "fetch_patient_context",
        "message": f"    ●   Looking up clinical records for patient {patient_id}...",
    })

    result = await db_reader.get_latest_patient_context(patient_id)

    if result is None:
        return _not_found("fetch_patient_context", patient_id, "patient_context")

    demo = result.get("demographics", {})
    name = demo.get("name", "Unknown")
    conditions = len(result.get("active_conditions", []))
    meds = len(result.get("medications", []))
    labs = len(result.get("lab_results", []))

    summary = f"Retrieved records for {name}"
    if conditions:
        summary += f" • {conditions} condition{'s' if conditions != 1 else ''}"
    if meds:
        summary += f" • {meds} medication{'s' if meds != 1 else ''}"
    if labs:
        summary += f" • {labs} lab result{'s' if labs != 1 else ''}"

    _emit({
        "type": "tool_complete",
        "tool": "fetch_patient_context",
        "message": f"   ✔   {summary}",
        "data": {
            "patient_name":      name,
            "patient_id":        patient_id,
            "conditions_count":  conditions,
            "medications_count": meds,
            "labs_count":        labs,
        },
    })

    return result


@tool
async def run_diagnosis(patient_id: str) -> dict:
    """
    Retrieve the most recent differential diagnosis for a patient.

    Returns top diagnosis with ICD-10 code, confidence level, full differential
    list, recommended next steps, and safety flags (penicillin allergy, sepsis
    risk, isolation requirements).

    When to use:
    - "What's the diagnosis?"
    - "What could be causing the symptoms?"
    - "Give me the differential for this patient"

    Args:
        patient_id: The patient's ID

    Returns:
        Differential diagnosis with ICD-10 codes, confidence, clinical reasoning,
        recommended next steps, and safety flags.
    """
    _emit({
        "type": "tool_start",
        "tool": "run_diagnosis",
        "message": f"    ●   Looking up stored diagnosis for patient {patient_id}...",
    })

    result = await db_reader.get_latest_diagnosis(patient_id)

    if result is None:
        return _not_found("run_diagnosis", patient_id, "diagnosis_results")

    top_dx = result.get("top_diagnosis", "Unknown")
    top_code = result.get("top_icd10_code", "")
    confidence = result.get("confidence_level", "")
    n_diff = len(result.get("differential_diagnosis", []))

    flags = []
    if result.get("penicillin_allergy_flagged"):
        flags.append("    ⚠   Penicillin allergy")
    if result.get("high_suspicion_sepsis"):
        flags.append("    ☣   Sepsis risk")
    if result.get("requires_isolation"):
        flags.append("    ⚠   Isolation required")

    flag_str = " | " + " | ".join(flags) if flags else ""

    _emit({
        "type": "tool_complete",
        "tool": "run_diagnosis",
        "message": f"   ✔   {top_dx} ({top_code}) — {confidence} confidence | {n_diff} differentials{flag_str}",
        "data": {
            "top_diagnosis":      top_dx,
            "top_icd10_code":     top_code,
            "confidence_level":   confidence,
            "differential_count": n_diff,
            "safety_flags":       flags,
        },
    })

    return result


@tool
async def analyze_labs(patient_id: str) -> dict:
    """
    Retrieve the most recent laboratory analysis for a patient.

    Returns flagged abnormal values, identified clinical patterns (e.g., SIRS
    criteria, metabolic acidosis), severity scoring, critical alerts, and
    clinical decision support recommendations.

    When to use:
    - "What do the labs show?"
    - "Are the labs normal?"
    - "Explain the abnormal lab results"
    - "What's the lab severity?"

    Args:
        patient_id: The patient's ID

    Returns:
        Lab summary with flagged results, pattern analysis, severity scoring,
        critical alerts, and clinical decision support.
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_labs",
        "message": f"    ●   Looking up stored lab analysis for patient {patient_id}...",
    })

    result = await db_reader.get_latest_lab_analysis(patient_id)

    if result is None:
        return _not_found("analyze_labs", patient_id, "lab_analysis_results")

    summary = result.get("lab_summary", {})
    critical = summary.get("critical_count", 0)
    abnormal = summary.get("abnormal_count", 0)
    severity = summary.get("overall_severity", "NORMAL")
    n_alerts = len(result.get("critical_alerts", []))

    severity_emoji = {
        "CRITICAL": "    ☣   ", "HIGH": "    ⚠   ", "MODERATE": "    ▶   ", "LOW": "    ℹ   ", "NORMAL": "   ✔  "
    }.get(severity, "")

    msg = f"{severity_emoji} Lab severity: {severity}"
    if critical:
        msg += f" | {critical} critical value{'s' if critical != 1 else ''}"
    if abnormal:
        msg += f" | {abnormal} abnormal"
    if n_alerts:
        msg += f" | {n_alerts} alert{'s' if n_alerts != 1 else ''}"

    _emit({
        "type": "tool_complete",
        "tool": "analyze_labs",
        "message": msg,
        "data": {
            "critical_count": critical,
            "abnormal_count": abnormal,
            "severity":       severity,
            "alerts_count":   n_alerts,
        },
    })

    return result


@tool
async def check_drug_safety(patient_id: str) -> dict:
    """
    Retrieve the most recent drug safety check for a patient.

    Returns safety status (SAFE / CAUTION / UNSAFE), flagged medications,
    contraindications, critical drug-drug interactions, FDA warnings, and
    approved safe alternatives.

    When to use:
    - "Is this drug safe for the patient?"
    - "Any drug interactions?"
    - "Check medication safety"
    - Before prescribing

    Args:
        patient_id: The patient's ID

    Returns:
        Safety status, flagged medications, critical interactions,
        contraindications, FDA warnings, and approved medications.
    """
    _emit({
        "type": "tool_start",
        "tool": "check_drug_safety",
        "message": f"    ●   Looking up stored drug safety check for patient {patient_id}...",
    })

    result = await db_reader.get_latest_drug_safety(patient_id)

    if result is None:
        return _not_found("check_drug_safety", patient_id, "drug_safety_results")

    status = result.get("safety_status", "UNKNOWN")
    flagged = result.get("flagged_medications", [])
    approved = result.get("approved_medications", [])
    interactions = len(result.get("critical_interactions", []))

    status_emoji = {"SAFE": "    ✔   ", "UNSAFE": "    ✘   ", "CAUTION": "    ⚠   "}.get(status, "")

    msg = f"{status_emoji} Safety status: {status}"
    if approved:
        msg += f" | {len(approved)} medication{'s' if len(approved) != 1 else ''} cleared"
    if flagged:
        msg += f" | {len(flagged)} flagged: {', '.join(str(f) for f in flagged[:2])}"
    if interactions:
        msg += f" | {interactions} interaction{'s' if interactions != 1 else ''} detected"

    _emit({
        "type": "tool_complete",
        "tool": "check_drug_safety",
        "message": msg,
        "data": {
            "safety_status":      status,
            "flagged":            flagged,
            "approved":           approved,
            "interactions_count": interactions,
        },
    })

    return result


@tool
async def analyze_chest_xray(patient_id: str) -> dict:
    """
    Retrieve the most recent chest X-ray analysis for a patient.

    Returns CNN prediction (PNEUMONIA vs NORMAL), confidence score, severity
    grading, triage priority, LLM-generated clinical interpretation, and
    recommended actions. Uses EfficientNetB0 (AUC 0.981).

    Only call this when:
    - The user explicitly mentions a chest X-ray or imaging
    - The patient has had imaging analysis done before

    Args:
        patient_id: The patient's ID

    Returns:
        Model prediction, severity assessment, imaging findings, clinical
        interpretation, and FHIR DiagnosticReport.
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_chest_xray",
        "message": f"    ●   Looking up stored imaging analysis for patient {patient_id}...",
    })

    result = await db_reader.get_latest_imaging(patient_id)

    if result is None:
        return _not_found("analyze_chest_xray", patient_id, "imaging_triage_results")

    prediction = result.get("model_output", {}).get("prediction", "UNKNOWN")
    confidence = result.get("model_output", {}).get("confidence", 0)
    triage = result.get("severity_assessment", {}).get("triage_label", "UNKNOWN")
    mock = result.get("mock", False)

    emoji = {"PNEUMONIA": "    🫁   ", "NORMAL": "    ✔   "}.get(prediction, "  ◈  ")
    mock_note = " [mock data]" if mock else ""

    _emit({
        "type": "tool_complete",
        "tool": "analyze_chest_xray",
        "message": f"{emoji} {prediction} ({confidence:.0%} confidence) | Triage: {triage}{mock_note}",
        "data": {
            "prediction": prediction,
            "confidence": confidence,
            "triage":     triage,
            "is_mock":    mock,
        },
    })

    return result


@tool
async def simulate_treatment_outcomes(patient_id: str) -> dict:
    """
    Retrieve the most recent Digital Twin treatment simulation for a patient.

    Returns XGBoost-based outcome predictions including patient risk profile,
    baseline risks (30-day mortality, readmission, complication), treatment
    scenario comparisons, recommended option, what-if narrative, and FHIR CarePlan.

    When to use:
    - "What are the treatment options?"
    - "Compare treatments"
    - "What's the prognosis?"
    - "Simulate outcomes"

    Args:
        patient_id: The patient's ID

    Returns:
        Risk profile, baseline risks, scenario comparisons, recommended option,
        what-if narrative, FHIR CarePlan, and cost-effectiveness analysis.
    """
    _emit({
        "type": "tool_start",
        "tool": "simulate_treatment_outcomes",
        "message": f"    ●   Looking up stored Digital Twin simulation for patient {patient_id}...",
    })

    result = await db_reader.get_latest_simulation(patient_id)

    if result is None:
        return _not_found("simulate_treatment_outcomes", patient_id, "digital_twin_simulations")

    summary = result.get("simulation_summary", {})
    risk_profile = (
        summary.get("patient_risk_profile")
        or result.get("patient_risk_profile", "UNKNOWN")
    )
    recommended = result.get("recommended_option", "?")
    rec_confidence = result.get("recommendation_confidence", 0)

    risk_emoji = {
        "CRITICAL": "    ☣   ", "HIGH": "    ⚠   ", "MEDIUM": "    ▶   ", "LOW": "    ℹ   ", "NORMAL": "    ✔   "
    }.get(str(risk_profile).upper(), "")

    _emit({
        "type": "tool_complete",
        "tool": "simulate_treatment_outcomes",
        "message": f"  ✔  Risk: {risk_emoji} {risk_profile} | Recommended: Option {recommended} ({rec_confidence:.0%} confidence)",
        "data": {
            "risk_profile":       risk_profile,
            "recommended_option": recommended,
            "confidence":         rec_confidence,
        },
    })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# HTTP TOOLS — read all inputs from DB, then POST for live computation
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def run_consensus(patient_id: str) -> dict:
    """
    Cross-validate all specialist agent outputs for a patient and resolve conflicts.

    Reads the latest stored results (diagnosis, labs, imaging, drug safety) from
    the database and submits them to the consensus engine which:
    - Detects conflicts between agent outputs
    - Resolves conflicts using medical knowledge (tiebreaker RAG)
    - Escalates to human review when conflicts are severe
    - Produces a final consensus diagnosis with aggregate confidence

    Use this after the patient has been processed through the orchestrator to
    get a validated unified assessment.

    When to use:
    - "Give me a consensus on this patient"
    - After reviewing diagnosis + labs + imaging
    - Before generating the final clinical report

    Args:
        patient_id: The patient's ID

    Returns:
        Consensus status (FULL_CONSENSUS / CONFLICT_RESOLVED / ESCALATION_REQUIRED),
        final diagnosis, aggregate confidence, conflict details, human review flag.
    """
    _emit({
        "type": "tool_start",
        "tool": "run_consensus",
        "message": f"   ●    Reading all stored results for patient {patient_id} and running consensus...",
    })

    patient_state, diagnosis, lab_output, imaging_output, drug_safety_output = (
        await db_reader.get_latest_patient_context(patient_id),
        await db_reader.get_latest_diagnosis(patient_id),
        await db_reader.get_latest_lab_analysis(patient_id),
        await db_reader.get_latest_imaging(patient_id),
        await db_reader.get_latest_drug_safety(patient_id),
    )

    if diagnosis is None:
        return _not_found("run_consensus", patient_id, "diagnosis_results")

    sources = ["diagnosis"]
    if lab_output:         sources.append("labs")
    if imaging_output:     sources.append("imaging")
    if drug_safety_output: sources.append("drug safety")

    _emit({
        "type": "tool_progress",
        "tool": "run_consensus",
        "message": f"Cross-validating {len(sources)} source{'s' if len(sources) != 1 else ''}: {', '.join(sources)}...",
    })

    result = await _post(
        "Consensus",
        f"{CONSENSUS_URL}/consensus",
        {
            "diagnosis_output":   diagnosis,
            "lab_output":         lab_output,
            "imaging_output":     imaging_output,
            "drug_safety_output": drug_safety_output,
            "patient_state":      patient_state or {},
        },
        timeout=45.0,
    )

    if "error" in result:
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ {result.get('user_message', 'Consensus failed')}"})
    else:
        status = result.get("consensus_status", "UNKNOWN")
        confidence = result.get("aggregate_confidence", 0)
        conflicts = result.get("conflict_count", 0)
        human_review = result.get("human_review_required", False)

        status_emoji = {
            "FULL_CONSENSUS":      "   ✔   ",
            "CONFLICT_RESOLVED":   "   ⚠   ",
            "ESCALATION_REQUIRED": "   ☣   ",
        }.get(status, "")

        msg = f"{status_emoji} Consensus: {status} | Confidence: {confidence:.0%}"
        if conflicts:
            msg += f" | {conflicts} conflict{'s' if conflicts != 1 else ''} detected"
        if human_review:
            msg += " |   ⚠   HUMAN REVIEW REQUIRED"

        _emit({
            "type": "tool_complete",
            "tool": "run_consensus",
            "message": msg,
            "data": {
                "consensus_status":      status,
                "confidence":            confidence,
                "conflicts":             conflicts,
                "human_review_required": human_review,
            },
        })

    return result


@tool
async def generate_clinical_report(patient_id: str, chief_complaint: str) -> dict:
    """
    Generate the complete clinical documentation package for a patient.

    Reads all stored results from the database, then generates:

    1. SOAP Note (for clinicians):
       - Subjective: Patient history and presenting complaint
       - Objective: Vital signs, labs, imaging, medications
       - Assessment: Diagnosis with confidence, risk profile, drug safety
       - Plan: Ordered clinical action list

    2. Patient Explanation (plain language, grade 6-8 reading level):
       - What's wrong and why it happened
       - What treatment will be done and what to expect

    3. Risk Attribution (SHAP-style breakdown):
       - Top factors contributing to readmission/complication risk

    4. FHIR R4 Bundle:
       - Condition, DiagnosticReport, MedicationRequest, CarePlan resources

    When to use:
    - "Generate a SOAP note"
    - "Create a clinical summary"
    - "I need the full documentation"

    Args:
        patient_id: The patient's ID
        chief_complaint: The patient's primary complaint (e.g., "cough and fever")

    Returns:
        SOAP note, patient explanation, risk attribution, FHIR bundle,
        reading level check, consensus status, and human review flag.
    """
    _emit({
        "type": "tool_start",
        "tool": "generate_clinical_report",
        "message": f"   ●    Reading all stored results for patient {patient_id} and generating clinical documentation...",
    })

    (
        patient_state,
        diagnosis,
        lab_output,
        imaging_output,
        drug_safety_output,
        digital_twin_output,
    ) = (
        await db_reader.get_latest_patient_context(patient_id),
        await db_reader.get_latest_diagnosis(patient_id),
        await db_reader.get_latest_lab_analysis(patient_id),
        await db_reader.get_latest_imaging(patient_id),
        await db_reader.get_latest_drug_safety(patient_id),
        await db_reader.get_latest_simulation(patient_id),
    )

    if diagnosis is None:
        return _not_found("generate_clinical_report", patient_id, "diagnosis_results")

    _emit({
        "type": "tool_progress",
        "tool": "generate_clinical_report",
        "message": "Writing SOAP note and patient explanation (using LLM)...",
    })

    result = await _post(
        "Explanation",
        f"{EXPLANATION_URL}/explain",
        {
            "patient_state":       patient_state or {},
            "diagnosis_output":    diagnosis,
            "lab_output":          lab_output,
            "imaging_output":      imaging_output,
            "drug_safety_output":  drug_safety_output,
            "digital_twin_output": digital_twin_output,
            "consensus_output":    None,
            "chief_complaint":     chief_complaint,
        },
        timeout=90.0,
    )

    if "error" in result:
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ {result.get('user_message', 'Report generation failed')}"})
    else:
        rl = result.get("reading_level_check", {})
        grade = rl.get("grade_level", "?")
        n_fhir = result.get("fhir_bundle", {}).get("_entry_count", 0)
        consensus_status = result.get("consensus_status", "UNKNOWN")

        _emit({
            "type": "tool_complete",
            "tool": "generate_clinical_report",
            "message": f"   ✔   SOAP note generated | Patient explanation (grade {grade}) | {n_fhir} FHIR resources | Status: {consensus_status}",
            "data": {
                "reading_grade":    grade,
                "fhir_resources":   n_fhir,
                "consensus_status": consensus_status,
            },
        })

    return result


# ── Tool registry ──────────────────────────────────────────────────────────────

MEDITWIN_TOOLS = [
    fetch_patient_context,
    run_diagnosis,
    analyze_labs,
    check_drug_safety,
    analyze_chest_xray,
    simulate_treatment_outcomes,
    run_consensus,
    generate_clinical_report,
]
