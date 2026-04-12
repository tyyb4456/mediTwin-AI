"""
MediTwin Tool Agent — Agent Tools
Each downstream specialist agent is exposed as a @tool using langchain_core.tools.

Streaming: Each tool emits custom stream events via get_stream_writer so the client
gets live progress updates as independent agents run (e.g. "🔬 Running lab analysis...").
Since specialist agents only depend on patient state, they are effectively independent
and can all emit progress without waiting for each other.
"""
import os
import json
import logging
import asyncio
import httpx
from typing import Optional, List
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

logger = logging.getLogger("meditwin.tool_agent.tools")

# ── Service URLs ──────────────────────────────────────────────────────────────
PATIENT_CONTEXT_URL = os.getenv("PATIENT_CONTEXT_URL",  "http://localhost:8001")
DIAGNOSIS_URL       = os.getenv("DIAGNOSIS_URL",         "http://localhost:8002")
LAB_ANALYSIS_URL    = os.getenv("LAB_ANALYSIS_URL",      "http://localhost:8003")
DRUG_SAFETY_URL     = os.getenv("DRUG_SAFETY_URL",       "http://localhost:8004")
IMAGING_TRIAGE_URL  = os.getenv("IMAGING_TRIAGE_URL",   "http://localhost:8005")
DIGITAL_TWIN_URL    = os.getenv("DIGITAL_TWIN_URL",      "http://localhost:8006")
CONSENSUS_URL       = os.getenv("CONSENSUS_URL",         "http://localhost:8007")
EXPLANATION_URL     = os.getenv("EXPLANATION_URL",       "http://localhost:8009")

INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
HEADERS = {
    "Authorization": f"Bearer {INTERNAL_TOKEN}",
    "Content-Type": "application/json",
}


# ── Async HTTP helper ─────────────────────────────────────────────────────────

async def _post(name: str, url: str, payload: dict, timeout: float = 45.0) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=HEADERS)
            r.raise_for_status()
            return json.dumps(r.json())
    except httpx.TimeoutException:
        logger.warning(f"[{name}] Timeout after {timeout}s")
        return json.dumps({"error": f"{name} timed out after {timeout}s"})
    except httpx.HTTPStatusError as e:
        logger.error(f"[{name}] HTTP {e.response.status_code}")
        return json.dumps({"error": f"{name} HTTP {e.response.status_code}", "detail": e.response.text[:300]})
    except Exception as e:
        logger.error(f"[{name}] Unexpected: {e}")
        return json.dumps({"error": f"{name} failed: {str(e)}"})


def _run(coro):
    """Bridge async tool calls into the sync @tool context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Stream writer helper ──────────────────────────────────────────────────────

def _emit(event: dict):
    """
    Emit a custom stream event. Called inside @tool functions.
    The client receives these in real-time via stream_mode="custom".

    Event shape:
        {
            "type":   "tool_start" | "tool_complete" | "tool_error" | "tool_progress",
            "tool":   "<tool_name>",
            "message": "<human readable status>",
            "data":   <optional payload>
        }
    """
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        # get_stream_writer raises if called outside LangGraph context (e.g. unit tests)
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Patient Context  (ALWAYS first when patient_id is known)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def fetch_patient_context(
    patient_id: str,
    fhir_base_url: str = "https://hapi.fhir.org/baseR4",
    sharp_token: Optional[str] = None,
) -> str:
    """
    Fetch the complete clinical profile of a patient from the FHIR server.

    WHEN TO USE:
      Call this tool first, immediately after extracting a patient_id from the user's
      query. Every other tool depends on the data this tool returns.
      Never call any other clinical tool before this one.

    RETURNS:
      A PatientState JSON containing demographics, active conditions (ICD-10),
      current medications with dosing, allergies, lab results with flags,
      diagnostic reports, and recent encounters.

    Args:
        patient_id: FHIR Patient resource ID extracted from the user query
        fhir_base_url: FHIR R4 server base URL (default: HAPI public sandbox)
        sharp_token: Optional SHARP bearer token for authenticated FHIR access
    """
    _emit({
        "type": "tool_start",
        "tool": "fetch_patient_context",
        "message": f"Fetching patient record for {patient_id}...",
    })

    result = _run(_post(
        "PatientContext",
        f"{PATIENT_CONTEXT_URL}/fetch",
        {"patient_id": patient_id, "fhir_base_url": fhir_base_url, "sharp_token": sharp_token},
        timeout=20.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({
            "type": "tool_error",
            "tool": "fetch_patient_context",
            "message": f"Failed to fetch patient {patient_id}: {parsed['error']}",
        })
    else:
        ps = parsed.get("patient_state", {})
        demo = ps.get("demographics", {})
        name = demo.get("name", "Unknown")
        conditions = len(ps.get("active_conditions", []))
        meds = len(ps.get("medications", []))
        labs = len(ps.get("lab_results", []))
        _emit({
            "type": "tool_complete",
            "tool": "fetch_patient_context",
            "message": f"Patient loaded: {name} | {conditions} conditions | {meds} meds | {labs} labs",
            "data": {"patient_id": patient_id, "cache_hit": parsed.get("cache_hit", False)},
        })

    # Return the patient_state directly so downstream tools receive it cleanly
    if "patient_state" in parsed:
        return json.dumps(parsed["patient_state"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

@tool
def run_diagnosis(patient_state_json: str, chief_complaint: str) -> str:
    """
    Generate a differential diagnosis using clinical AI reasoning and medical RAG.

    WHEN TO USE:
      When the user asks about: diagnosis, what condition the patient has,
      differential, ICD-10 codes, clinical reasoning, what is wrong with the patient,
      or requests a full clinical assessment.

    RETURNS:
      Ranked differential diagnosis list with ICD-10 codes, confidence scores,
      clinical reasoning, supporting/against evidence, recommended next steps,
      and safety flags (sepsis suspicion, penicillin allergy, isolation required).

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        chief_complaint: The patient's primary symptom or reason for visit
    """
    _emit({
        "type": "tool_start",
        "tool": "run_diagnosis",
        "message": f"Running differential diagnosis for: {chief_complaint[:60]}...",
    })

    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError:
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": "❌ Invalid patient state JSON"})
        return json.dumps({"error": "patient_state_json must be valid JSON"})

    result = _run(_post(
        "Diagnosis",
        f"{DIAGNOSIS_URL}/diagnose",
        {"patient_state": state, "chief_complaint": chief_complaint, "include_fhir_resources": True},
        timeout=60.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": f"❌ Diagnosis failed: {parsed['error']}"})
    else:
        top_dx = parsed.get("top_diagnosis", "Unknown")
        top_code = parsed.get("top_icd10_code", "")
        confidence = parsed.get("confidence_level", "")
        n_diff = len(parsed.get("differential_diagnosis", []))
        flags = []
        if parsed.get("penicillin_allergy_flagged"):
            flags.append("Penicillin allergy")
        if parsed.get("high_suspicion_sepsis"):
            flags.append("Sepsis risk")
        flag_str = " | ".join(flags) if flags else ""
        _emit({
            "type": "tool_complete",
            "tool": "run_diagnosis",
            "message": f"Diagnosis: {top_dx} ({top_code}) [{confidence}] | {n_diff} differentials{' | ' + flag_str if flag_str else ''}",
            "data": {"top_diagnosis": top_dx, "top_icd10_code": top_code, "confidence_level": confidence},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Lab Analysis
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_labs(patient_state_json: str, diagnosis_json: Optional[str] = None) -> str:
    """
    Perform deep clinical interpretation of the patient's laboratory results.

    WHEN TO USE:
      When the user asks about: lab results, blood work, abnormal values, WBC, CBC,
      metabolic panel, lab trends, critical values, or any specific lab test interpretation.
      Also use when running a full assessment where labs are available in patient context.

    RETURNS:
      Flagged abnormal results with clinical significance, pattern analysis
      (e.g. inflammatory pattern, renal impairment), critical alerts,
      and confirmation or challenge of the working diagnosis.

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        diagnosis_json: Optional raw JSON string from run_diagnosis for richer context
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_labs",
        "message": "Analyzing laboratory results...",
    })

    try:
        state = json.loads(patient_state_json)
        diag = json.loads(diagnosis_json) if diagnosis_json else None
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": f"❌ Invalid JSON: {e}"})
        return json.dumps({"error": f"Invalid JSON: {e}"})

    n_labs = len(state.get("lab_results", []))
    _emit({
        "type": "tool_progress",
        "tool": "analyze_labs",
        "message": f"Processing {n_labs} lab results...",
    })

    result = _run(_post(
        "LabAnalysis",
        f"{LAB_ANALYSIS_URL}/analyze-labs",
        {"patient_state": state, "diagnosis_agent_output": diag},
        timeout=45.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": f"❌ Lab analysis failed: {parsed['error']}"})
    else:
        summary = parsed.get("lab_summary", {})
        critical = summary.get("critical_count", 0)
        abnormal = summary.get("abnormal_count", 0)
        severity = summary.get("overall_severity", "NORMAL")
        n_alerts = len(parsed.get("critical_alerts", []))
        _emit({
            "type": "tool_complete",
            "tool": "analyze_labs",
            "message": f"Labs: {severity} severity | {critical} critical | {abnormal} abnormal | {n_alerts} alerts",
            "data": {"critical_count": critical, "severity": severity},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Drug Safety
# ══════════════════════════════════════════════════════════════════════════════

@tool
def check_drug_safety(patient_state_json: str, proposed_medications: List[str]) -> str:
    """
    Check medication safety against the patient's allergies, current drugs, and conditions.

    WHEN TO USE:
      When the user asks about: medications, prescribing, drug interactions, allergy check,
      is it safe to give [drug] to this patient, drug contraindications, alternatives to a drug,
      or any question involving adding or changing medications for a specific patient.

    RETURNS:
      Safety status (SAFE / UNSAFE), critical drug-drug interactions, contraindications,
      allergy cross-reactivity warnings, approved safe medications, and safer alternatives.

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        proposed_medications: List of medication name strings to check,
                              e.g. ["Amoxicillin 500mg", "Azithromycin 500mg"]
    """
    med_list = ", ".join(proposed_medications)
    _emit({
        "type": "tool_start",
        "tool": "check_drug_safety",
        "message": f"Checking safety for: {med_list}...",
    })

    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": f"❌ Invalid patient state: {e}"})
        return json.dumps({"error": f"Invalid patient_state_json: {e}"})

    current_meds = [m.get("drug", "") for m in state.get("medications", [])]
    allergies    = state.get("allergies", [])
    conditions   = state.get("active_conditions", [])

    _emit({
        "type": "tool_progress",
        "tool": "check_drug_safety",
        "message": "Checking allergy cross-reactivity and drug interactions...",
    })

    result = _run(_post(
        "DrugSafety",
        f"{DRUG_SAFETY_URL}/check-safety",
        {
            "proposed_medications": proposed_medications,
            "current_medications":  current_meds,
            "patient_allergies":    allergies,
            "active_conditions":    conditions,
            "patient_id":           state.get("patient_id", "unknown"),
        },
        timeout=30.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": f"❌ Drug safety check failed: {parsed['error']}"})
    else:
        status = parsed.get("safety_status", "UNKNOWN")
        flagged = parsed.get("flagged_medications", [])
        approved = parsed.get("approved_medications", [])
        interactions = len(parsed.get("critical_interactions", []))
        flagged_str = f" | Flagged: {', '.join(flagged)}" if flagged else ""
        _emit({
            "type": "tool_complete",
            "tool": "check_drug_safety",
            "message": f"Drug safety: {status} | {len(approved)} approved | {interactions} interactions{flagged_str}",
            "data": {"safety_status": status, "flagged": flagged, "approved": approved},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Imaging Triage
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_chest_xray(patient_state_json: str, image_data_base64: str) -> str:
    """
    Analyze a chest X-ray image using CNN-based medical imaging AI.

    WHEN TO USE:
      ONLY when the user explicitly provides or references a chest X-ray image
      AND base64-encoded image data is present in the conversation.
      Do NOT call this tool if no image data is available.

    RETURNS:
      CNN prediction (e.g. PNEUMONIA, NORMAL), confidence score, severity assessment,
      imaging findings (consolidation, effusion, infiltrates), and clinical interpretation.

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        image_data_base64: Base64-encoded JPEG chest X-ray string
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_chest_xray",
        "message": "Analyzing chest X-ray with CNN model (EfficientNetB0)...",
    })

    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": f"❌ Invalid patient state: {e}"})
        return json.dumps({"error": f"Invalid patient_state_json: {e}"})

    result = _run(_post(
        "ImagingTriage",
        f"{IMAGING_TRIAGE_URL}/analyze-xray",
        {
            "patient_id": state.get("patient_id", "unknown"),
            "image_data": {"format": "base64", "content_type": "image/jpeg", "data": image_data_base64},
            "patient_context": {
                "age":    state.get("demographics", {}).get("age", 40),
                "gender": state.get("demographics", {}).get("gender", "unknown"),
            },
        },
        timeout=30.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": f"❌ Imaging failed: {parsed['error']}"})
    else:
        prediction = parsed.get("model_output", {}).get("prediction", "UNKNOWN")
        confidence = parsed.get("model_output", {}).get("confidence", 0)
        triage = parsed.get("severity_assessment", {}).get("triage_label", "UNKNOWN")
        mock = parsed.get("mock", False)
        mock_str = " [MOCK - no model loaded]" if mock else ""
        _emit({
            "type": "tool_complete",
            "tool": "analyze_chest_xray",
            "message": f"Imaging: {prediction} ({confidence:.0%} confidence) | Triage: {triage}{mock_str}",
            "data": {"prediction": prediction, "confidence": confidence, "triage": triage},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Digital Twin
# ══════════════════════════════════════════════════════════════════════════════

@tool
def simulate_treatment_outcomes(
    patient_state_json: str,
    diagnosis_json: str,
    treatment_options: List[dict],
) -> str:
    """
    Simulate patient outcomes under different treatment scenarios using a Digital Twin model.

    WHEN TO USE:
      When the user asks about: treatment outcomes, prognosis, what happens if we give X,
      comparing treatment options, recovery time, complication risk,
      hospitalization vs outpatient, or any forward-looking scenario question.

    RETURNS:
      Patient risk profile (LOW/MEDIUM/HIGH/CRITICAL), outcome predictions per option,
      recommended treatment with justification, and a what-if narrative.

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        diagnosis_json: Raw JSON string from run_diagnosis
        treatment_options: List of option dicts with keys:
                           option_id (str), label (str), drugs (List[str]), interventions (List[str])
    """
    n_options = len(treatment_options)
    _emit({
        "type": "tool_start",
        "tool": "simulate_treatment_outcomes",
        "message": f"🤖Running Digital Twin simulation for {n_options} treatment option(s)...",
    })

    try:
        state = json.loads(patient_state_json)
        diag  = json.loads(diagnosis_json)
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ Invalid JSON: {e}"})
        return json.dumps({"error": f"Invalid JSON: {e}"})

    option_labels = [o.get("label", o.get("option_id", "?")) for o in treatment_options]
    _emit({
        "type": "tool_progress",
        "tool": "simulate_treatment_outcomes",
        "message": f"🤖 Simulating: {', '.join(option_labels)}...",
    })

    diagnosis_str = f"{diag.get('top_diagnosis', 'Unknown')} ({diag.get('top_icd10_code', '')})"

    result = _run(_post(
        "DigitalTwin",
        f"{DIGITAL_TWIN_URL}/simulate",
        {"patient_state": state, "diagnosis": diagnosis_str, "treatment_options": treatment_options},
        timeout=45.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ Simulation failed: {parsed['error']}"})
    else:
        summary = parsed.get("simulation_summary", {})
        risk_profile = summary.get("patient_risk_profile", "UNKNOWN")
        recommended = summary.get("recommended_option", "?")
        rec_confidence = summary.get("recommendation_confidence", 0)
        _emit({
            "type": "tool_complete",
            "tool": "simulate_treatment_outcomes",
            "message": f"✅ Simulation complete | Risk: {risk_profile} | Recommended: Option {recommended} ({rec_confidence:.0%} confidence)",
            "data": {"risk_profile": risk_profile, "recommended_option": recommended},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — Consensus
# ══════════════════════════════════════════════════════════════════════════════

@tool
def run_consensus(
    patient_state_json: str,
    diagnosis_json: str,
    lab_json: Optional[str] = None,
    imaging_json: Optional[str] = None,
    drug_safety_json: Optional[str] = None,
) -> str:
    """
    Cross-validate all specialist agent outputs and produce a final clinical consensus.

    WHEN TO USE:
      After gathering outputs from multiple tools (diagnosis, labs, drug safety)
      and the user wants a final validated assessment, or when completing a full workup.
      Run this before generate_clinical_report.

    RETURNS:
      Consensus status (FULL_CONSENSUS / CONFLICT_RESOLVED / ESCALATION_REQUIRED),
      final agreed diagnosis with aggregate confidence, conflict results,
      and a human review flag when confidence is low.

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        diagnosis_json: Raw JSON string from run_diagnosis (required)
        lab_json: Optional raw JSON string from analyze_labs
        imaging_json: Optional raw JSON string from analyze_chest_xray
        drug_safety_json: Optional raw JSON string from check_drug_safety
    """
    sources = ["diagnosis"]
    if lab_json: sources.append("labs")
    if imaging_json: sources.append("imaging")
    if drug_safety_json: sources.append("drug safety")

    _emit({
        "type": "tool_start",
        "tool": "run_consensus",
        "message": f"⚖️ Running consensus across {len(sources)} agent(s): {', '.join(sources)}...",
    })

    try:
        state = json.loads(patient_state_json)
        diag  = json.loads(diagnosis_json)
        lab   = json.loads(lab_json)         if lab_json         else None
        img   = json.loads(imaging_json)     if imaging_json     else None
        drug  = json.loads(drug_safety_json) if drug_safety_json else None
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ Invalid JSON: {e}"})
        return json.dumps({"error": f"Invalid JSON: {e}"})

    result = _run(_post(
        "Consensus",
        f"{CONSENSUS_URL}/consensus",
        {
            "diagnosis_output":   diag,
            "lab_output":         lab,
            "imaging_output":     img,
            "drug_safety_output": drug,
            "patient_state":      state,
        },
        timeout=45.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ Consensus failed: {parsed['error']}"})
    else:
        status = parsed.get("consensus_status", "UNKNOWN")
        confidence = parsed.get("aggregate_confidence", 0)
        conflicts = parsed.get("conflict_count", 0)
        human_review = parsed.get("human_review_required", False)
        status_emoji = "✅" if status == "FULL_CONSENSUS" else ("⚠️" if status == "CONFLICT_RESOLVED" else "🚨")
        review_str = " | ⚠️ HUMAN REVIEW REQUIRED" if human_review else ""
        _emit({
            "type": "tool_complete",
            "tool": "run_consensus",
            "message": f"{status_emoji} Consensus: {status} | Confidence: {confidence:.0%} | {conflicts} conflicts{review_str}",
            "data": {"consensus_status": status, "confidence": confidence, "human_review_required": human_review},
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — Clinical Report / Explanation
# ══════════════════════════════════════════════════════════════════════════════

@tool
def generate_clinical_report(
    patient_state_json: str,
    consensus_json: str,
    chief_complaint: str,
    diagnosis_json: Optional[str] = None,
    lab_json: Optional[str] = None,
    imaging_json: Optional[str] = None,
    drug_safety_json: Optional[str] = None,
    digital_twin_json: Optional[str] = None,
) -> str:
    """
    Generate the final clinical documentation package for the patient.

    WHEN TO USE:
      When the user asks for: a report, SOAP note, patient summary, clinical letter,
      FHIR bundle, discharge summary, patient-friendly explanation, or a complete
      clinical write-up. Also use as the final step after a full workup.

    RETURNS:
      - SOAP note for clinicians (Subjective / Objective / Assessment / Plan)
      - Plain-language patient explanation (grade 6-8 reading level)
      - SHAP-style risk attribution breakdown
      - FHIR R4 Bundle ready for EHR submission

    Args:
        patient_state_json: Raw JSON string from fetch_patient_context
        consensus_json: Raw JSON string from run_consensus (required)
        chief_complaint: Patient's primary complaint
        diagnosis_json: Optional raw JSON string from run_diagnosis
        lab_json: Optional raw JSON string from analyze_labs
        imaging_json: Optional raw JSON string from analyze_chest_xray
        drug_safety_json: Optional raw JSON string from check_drug_safety
        digital_twin_json: Optional raw JSON string from simulate_treatment_outcomes
    """
    _emit({
        "type": "tool_start",
        "tool": "generate_clinical_report",
        "message": "📋 Generating clinical report (SOAP note, patient explanation, FHIR bundle)...",
    })

    try:
        state     = json.loads(patient_state_json)
        consensus = json.loads(consensus_json)
        diag      = json.loads(diagnosis_json)    if diagnosis_json    else None
        lab       = json.loads(lab_json)          if lab_json          else None
        img       = json.loads(imaging_json)      if imaging_json      else None
        drug      = json.loads(drug_safety_json)  if drug_safety_json  else None
        twin      = json.loads(digital_twin_json) if digital_twin_json else None
    except json.JSONDecodeError as e:
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ Invalid JSON: {e}"})
        return json.dumps({"error": f"Invalid JSON: {e}"})

    _emit({
        "type": "tool_progress",
        "tool": "generate_clinical_report",
        "message": "📋 Writing SOAP note and patient explanation (LLM)...",
    })

    result = _run(_post(
        "Explanation",
        f"{EXPLANATION_URL}/explain",
        {
            "patient_state":       state,
            "consensus_output":    consensus,
            "diagnosis_output":    diag,
            "lab_output":          lab,
            "imaging_output":      img,
            "drug_safety_output":  drug,
            "digital_twin_output": twin,
            "chief_complaint":     chief_complaint,
        },
        timeout=90.0,
    ))

    parsed = json.loads(result)
    if "error" in parsed:
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ Report generation failed: {parsed['error']}"})
    else:
        rl = parsed.get("reading_level_check", {})
        grade = rl.get("grade_level", "?")
        n_fhir = parsed.get("fhir_bundle", {}).get("_entry_count", 0)
        consensus_status = parsed.get("consensus_status", "UNKNOWN")
        _emit({
            "type": "tool_complete",
            "tool": "generate_clinical_report",
            "message": f"✅ Report ready | SOAP note generated | Patient explanation grade {grade} | {n_fhir} FHIR resources | Status: {consensus_status}",
            "data": {"reading_grade": grade, "fhir_resources": n_fhir, "consensus_status": consensus_status},
        })

    return result


# ── Tool registry (imported by agent.py) ──────────────────────────────────────
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