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


# ── Patient state normalization ───────────────────────────────────────────────
# LangGraph serializes tool return values as JSON in ToolMessage.content.
# When the LLM passes that result to the next tool, it sometimes wraps it in
# {"fetch_patient_context_response": {...}} or passes a JSON string instead of
# a dict. This helper unwraps all those cases and always returns a plain dict.

def _unwrap_patient_state(raw) -> dict:
    """
    Normalize whatever the LLM passes as patient_state into a clean dict.
    Handles:
      - Already a correct dict with patient_id → return as-is
      - JSON string → parse it, then recurse
      - LangGraph wrapper {"fetch_patient_context_response": {...}} → unwrap
      - Any other single-key wrapper dict → unwrap the value
    """
    # Case 1: JSON string — parse and recurse
    if isinstance(raw, str):
        try:
            return _unwrap_patient_state(json.loads(raw))
        except json.JSONDecodeError:
            logger.error(f"patient_state is an unparseable string: {raw[:100]}")
            return {}

    if not isinstance(raw, dict):
        logger.error(f"patient_state is unexpected type: {type(raw)}")
        return {}

    # Case 2: Already a valid PatientState dict
    if "patient_id" in raw:
        return raw

    # Case 3: LangGraph wrapper — single key whose value contains the state
    # e.g. {"fetch_patient_context_response": {...}}
    if len(raw) == 1:
        inner = next(iter(raw.values()))
        return _unwrap_patient_state(inner)

    # Case 4: Multi-key dict but no patient_id — log and return as-is
    logger.warning(f"patient_state has no patient_id, keys: {list(raw.keys())[:5]}")
    return raw


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

async def _post(name: str, url: str, payload: dict, timeout: float = 45.0) -> dict:
    """Returns dict, not JSON string—LangChain handles serialization."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=HEADERS)
            r.raise_for_status()
            return r.json()  # Return dict
    except httpx.TimeoutException:
        logger.warning(f"[{name}] Timeout after {timeout}s")
        return {"error": f"{name} timed out after {timeout}s"}
    except httpx.HTTPStatusError as e:
        logger.error(f"[{name}] HTTP {e.response.status_code}")
        return {"error": f"{name} HTTP {e.response.status_code}", "detail": e.response.text[:300]}
    except Exception as e:
        logger.error(f"[{name}] Unexpected: {e}")
        return {"error": f"{name} failed: {str(e)}"}
 
 
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
    """Emit a custom stream event."""
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Patient Context  (ALWAYS first when patient_id is known)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def fetch_patient_context(
    patient_id: str,
    fhir_base_url: str = "https://hapi.fhir.org/baseR4",
    sharp_token: Optional[str] = None,
) -> dict:
    """
    Fetch the complete clinical profile of a patient from the FHIR server.
 
    WHEN TO USE:
      Call this tool first, immediately after extracting a patient_id from the user's
      query. Every other tool depends on the data this tool returns.
      Never call any other clinical tool before this one.
 
    RETURNS:
      A PatientState dict containing demographics, active conditions (ICD-10),
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
 
    if "error" in result:
        _emit({
            "type": "tool_error",
            "tool": "fetch_patient_context",
            "message": f"Failed to fetch patient {patient_id}: {result['error']}",
        })
        return result  # Return dict with error
 
    ps = result.get("patient_state", {})
    demo = ps.get("demographics", {})
    name = demo.get("name", "Unknown")
    conditions = len(ps.get("active_conditions", []))
    meds = len(ps.get("medications", []))
    labs = len(ps.get("lab_results", []))
    _emit({
        "type": "tool_complete",
        "tool": "fetch_patient_context",
        "message": f"Patient loaded: {name} | {conditions} conditions | {meds} meds | {labs} labs",
        "data": {"patient_id": patient_id, "cache_hit": result.get("cache_hit", False)},
    })
 
    # CRITICAL FIX: Return the patient_state dict directly, not json.dumps()
    return ps

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

@tool
def run_diagnosis(patient_state: dict, chief_complaint: str) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        chief_complaint: The patient's primary symptom or reason for visit
    """
    _emit({
        "type": "tool_start",
        "tool": "run_diagnosis",
        "message": f"Running differential diagnosis for: {chief_complaint[:60]}...",
    })
 
    patient_state = _unwrap_patient_state(patient_state)
    if not patient_state.get("patient_id"):
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": "❌ patient_state missing patient_id after unwrap"})
        return {"error": "patient_state is missing patient_id — ensure fetch_patient_context ran first"}

    result = _run(_post(
        "Diagnosis",
        f"{DIAGNOSIS_URL}/diagnose",
        {"patient_state": patient_state, "chief_complaint": chief_complaint, "include_fhir_resources": True},
        timeout=60.0,
    ))
 
    if "error" in result:
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": f"❌ Diagnosis failed: {result['error']}"})
    else:
        top_dx = result.get("top_diagnosis", "Unknown")
        top_code = result.get("top_icd10_code", "")
        confidence = result.get("confidence_level", "")
        n_diff = len(result.get("differential_diagnosis", []))
        flags = []
        if result.get("penicillin_allergy_flagged"):
            flags.append("Penicillin allergy")
        if result.get("high_suspicion_sepsis"):
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
def analyze_labs(patient_state: dict, diagnosis: Optional[dict] = None) -> dict:
    """
    Perform deep clinical interpretation of the patient's laboratory results.
 
    WHEN TO USE:
      When the user asks about: lab results, blood work, abnormal values, WBC, CBC,
      metabolic panel, lab trends, critical values, or any specific lab test interpretation.
 
    Args:
        patient_state: Dict from fetch_patient_context
        diagnosis: Optional dict from run_diagnosis for richer context
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_labs",
        "message": "Analyzing laboratory results...",
    })
 
    patient_state = _unwrap_patient_state(patient_state)
    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": "❌ Invalid patient_state"})
        return {"error": "patient_state must be a dict"}
 
    n_labs = len(patient_state.get("lab_results", []))
    _emit({
        "type": "tool_progress",
        "tool": "analyze_labs",
        "message": f"Processing {n_labs} lab results...",
    })
 
    result = _run(_post(
        "LabAnalysis",
        f"{LAB_ANALYSIS_URL}/analyze-labs",
        {"patient_state": patient_state, "diagnosis_agent_output": diagnosis},
        timeout=45.0,
    ))
 
    if "error" in result:
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": f"❌ Lab analysis failed: {result['error']}"})
    else:
        summary = result.get("lab_summary", {})
        critical = summary.get("critical_count", 0)
        abnormal = summary.get("abnormal_count", 0)
        severity = summary.get("overall_severity", "NORMAL")
        n_alerts = len(result.get("critical_alerts", []))
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
def check_drug_safety(patient_state: dict, proposed_medications: List[str]) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        proposed_medications: List of medication name strings to check,
                              e.g. ["Amoxicillin 500mg", "Azithromycin 500mg"]
    """
    med_list = ", ".join(proposed_medications)
    _emit({
        "type": "tool_start",
        "tool": "check_drug_safety",
        "message": f"Checking safety for: {med_list}...",
    })

    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": "❌ patient_state must be a dict"})
        return {"error": "patient_state must be a dict"}

    state = _unwrap_patient_state(patient_state)

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

    if "error" in result:
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": f"❌ Drug safety check failed: {result['error']}"})
    else:
        status = result.get("safety_status", "UNKNOWN")
        flagged = result.get("flagged_medications", [])
        approved = result.get("approved_medications", [])
        interactions = len(result.get("critical_interactions", []))
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
def analyze_chest_xray(patient_state: dict, image_data_base64: str) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        image_data_base64: Base64-encoded JPEG chest X-ray string
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_chest_xray",
        "message": "Analyzing chest X-ray with CNN model (EfficientNetB0)...",
    })

    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": "❌ patient_state must be a dict"})
        return {"error": "patient_state must be a dict"}

    state = _unwrap_patient_state(patient_state)

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

    if "error" in result:
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": f"❌ Imaging failed: {result['error']}"})
    else:
        prediction = result.get("model_output", {}).get("prediction", "UNKNOWN")
        confidence = result.get("model_output", {}).get("confidence", 0)
        triage = result.get("severity_assessment", {}).get("triage_label", "UNKNOWN")
        mock = result.get("mock", False)
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
    patient_state: dict,
    diagnosis: dict,
    treatment_options: List[dict],
) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        diagnosis: Dict from run_diagnosis (pass as-is)
        treatment_options: List of option dicts with keys:
                           option_id (str), label (str), drugs (List[str]), interventions (List[str])
    """
    n_options = len(treatment_options)
    _emit({
        "type": "tool_start",
        "tool": "simulate_treatment_outcomes",
        "message": f"🤖Running Digital Twin simulation for {n_options} treatment option(s)...",
    })

    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": "❌ patient_state must be a dict"})
        return {"error": "patient_state must be a dict"}
    if not isinstance(diagnosis, dict):
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": "❌ diagnosis must be a dict"})
        return {"error": "diagnosis must be a dict"}

    state = _unwrap_patient_state(patient_state)
    diag = diagnosis

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

    if "error" in result:
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ Simulation failed: {result['error']}"})
    else:
        summary = result.get("simulation_summary", {})
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
    patient_state: dict,
    diagnosis: dict,
    lab_output: Optional[dict] = None,
    imaging_output: Optional[dict] = None,
    drug_safety_output: Optional[dict] = None,
) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        diagnosis: Dict from run_diagnosis (pass as-is, required)
        lab_output: Optional dict from analyze_labs (pass as-is)
        imaging_output: Optional dict from analyze_chest_xray (pass as-is)
        drug_safety_output: Optional dict from check_drug_safety (pass as-is)
    """
    sources = ["diagnosis"]
    if lab_output: sources.append("labs")
    if imaging_output: sources.append("imaging")
    if drug_safety_output: sources.append("drug safety")

    _emit({
        "type": "tool_start",
        "tool": "run_consensus",
        "message": f"⚖️ Running consensus across {len(sources)} agent(s): {', '.join(sources)}...",
    })

    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "run_consensus", "message": "❌ patient_state must be a dict"})
        return {"error": "patient_state must be a dict"}
    if not isinstance(diagnosis, dict):
        _emit({"type": "tool_error", "tool": "run_consensus", "message": "❌ diagnosis must be a dict"})
        return {"error": "diagnosis must be a dict"}

    patient_state = _unwrap_patient_state(patient_state)

    result = _run(_post(
        "Consensus",
        f"{CONSENSUS_URL}/consensus",
        {
            "diagnosis_output":   diagnosis,
            "lab_output":         lab_output,
            "imaging_output":     imaging_output,
            "drug_safety_output": drug_safety_output,
            "patient_state":      patient_state,
        },
        timeout=45.0,
    ))

    if "error" in result:
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ Consensus failed: {result['error']}"})
    else:
        status = result.get("consensus_status", "UNKNOWN")
        confidence = result.get("aggregate_confidence", 0)
        conflicts = result.get("conflict_count", 0)
        human_review = result.get("human_review_required", False)
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
    patient_state: dict,
    consensus: dict,
    chief_complaint: str,
    diagnosis: Optional[dict] = None,
    lab_output: Optional[dict] = None,
    imaging_output: Optional[dict] = None,
    drug_safety_output: Optional[dict] = None,
    digital_twin_output: Optional[dict] = None,
) -> dict:
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
        patient_state: Dict from fetch_patient_context (pass as-is)
        consensus: Dict from run_consensus (pass as-is, required)
        chief_complaint: Patient's primary complaint
        diagnosis: Optional dict from run_diagnosis (pass as-is)
        lab_output: Optional dict from analyze_labs (pass as-is)
        imaging_output: Optional dict from analyze_chest_xray (pass as-is)
        drug_safety_output: Optional dict from check_drug_safety (pass as-is)
        digital_twin_output: Optional dict from simulate_treatment_outcomes (pass as-is)
    """
    _emit({
        "type": "tool_start",
        "tool": "generate_clinical_report",
        "message": "📋 Generating clinical report (SOAP note, patient explanation, FHIR bundle)...",
    })

    if not isinstance(patient_state, dict):
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": "❌ patient_state must be a dict"})
        return {"error": "patient_state must be a dict"}
    if not isinstance(consensus, dict):
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": "❌ consensus must be a dict"})
        return {"error": "consensus must be a dict"}

    patient_state = _unwrap_patient_state(patient_state)

    _emit({
        "type": "tool_progress",
        "tool": "generate_clinical_report",
        "message": "📋 Writing SOAP note and patient explanation (LLM)...",
    })

    result = _run(_post(
        "Explanation",
        f"{EXPLANATION_URL}/explain",
        {
            "patient_state":       patient_state,
            "consensus_output":    consensus,
            "diagnosis_output":    diagnosis,
            "lab_output":          lab_output,
            "imaging_output":      imaging_output,
            "drug_safety_output":  drug_safety_output,
            "digital_twin_output": digital_twin_output,
            "chief_complaint":     chief_complaint,
        },
        timeout=90.0,
    ))

    if "error" in result:
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ Report generation failed: {result['error']}"})
    else:
        rl = result.get("reading_level_check", {})
        grade = rl.get("grade_level", "?")
        n_fhir = result.get("fhir_bundle", {}).get("_entry_count", 0)
        consensus_status = result.get("consensus_status", "UNKNOWN")
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