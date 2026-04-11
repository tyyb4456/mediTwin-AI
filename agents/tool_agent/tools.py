"""
MediTwin Tool Agent — Agent Tools
Each downstream specialist agent is exposed as a @tool using langchain_core.tools.

Design rules:
  - Every tool calls the corresponding agent's HTTP endpoint via httpx (async)
  - Docstrings teach the LLM WHEN to use each tool — be precise
  - Tools accept typed Python args; JSON serialization handled internally
  - All tools return a JSON string — the LLM reads this as an observation
  - Failures return {"error": "..."} JSON — never raise, LLM handles gracefully
"""
import os
import json
import logging
import asyncio
import httpx
from typing import Optional, List
from langchain_core.tools import tool

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
    return _run(_post(
        "PatientContext",
        f"{PATIENT_CONTEXT_URL}/fetch",
        {"patient_id": patient_id, "fhir_base_url": fhir_base_url, "sharp_token": sharp_token},
        timeout=20.0,
    ))


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
    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "patient_state_json must be valid JSON"})

    return _run(_post(
        "Diagnosis",
        f"{DIAGNOSIS_URL}/diagnose",
        {"patient_state": state, "chief_complaint": chief_complaint, "include_fhir_resources": True},
        timeout=60.0,
    ))


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
    try:
        state = json.loads(patient_state_json)
        diag = json.loads(diagnosis_json) if diagnosis_json else None
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    return _run(_post(
        "LabAnalysis",
        f"{LAB_ANALYSIS_URL}/analyze-labs",
        {"patient_state": state, "diagnosis_agent_output": diag},
        timeout=45.0,
    ))


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
    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid patient_state_json: {e}"})

    current_meds = [m.get("drug", "") for m in state.get("medications", [])]
    allergies    = state.get("allergies", [])
    conditions   = state.get("active_conditions", [])

    return _run(_post(
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
    try:
        state = json.loads(patient_state_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid patient_state_json: {e}"})

    return _run(_post(
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
    try:
        state = json.loads(patient_state_json)
        diag  = json.loads(diagnosis_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    diagnosis_str = f"{diag.get('top_diagnosis', 'Unknown')} ({diag.get('top_icd10_code', '')})"

    return _run(_post(
        "DigitalTwin",
        f"{DIGITAL_TWIN_URL}/simulate",
        {"patient_state": state, "diagnosis": diagnosis_str, "treatment_options": treatment_options},
        timeout=45.0,
    ))


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
    try:
        state = json.loads(patient_state_json)
        diag  = json.loads(diagnosis_json)
        lab   = json.loads(lab_json)         if lab_json         else None
        img   = json.loads(imaging_json)     if imaging_json     else None
        drug  = json.loads(drug_safety_json) if drug_safety_json else None
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    return _run(_post(
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
    try:
        state     = json.loads(patient_state_json)
        consensus = json.loads(consensus_json)
        diag      = json.loads(diagnosis_json)    if diagnosis_json    else None
        lab       = json.loads(lab_json)          if lab_json          else None
        img       = json.loads(imaging_json)      if imaging_json      else None
        drug      = json.loads(drug_safety_json)  if drug_safety_json  else None
        twin      = json.loads(digital_twin_json) if digital_twin_json else None
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    return _run(_post(
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