"""
Agent Callers — Orchestrator
Async HTTP wrappers for every downstream agent service.

Design rules (from strategy doc):
  - safe_call() wraps every call: no single agent failure crashes the system
  - Only Patient Context failure is fatal
  - All others return None on failure and log to error_log
  - Timeouts: 30s default, 60s for LLM-heavy agents (Diagnosis, Explanation)
"""
import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger("meditwin.orchestrator")

# ── Service URLs (from environment, with localhost defaults) ─────────────────
PATIENT_CONTEXT_URL = os.getenv("PATIENT_CONTEXT_URL",  "http://localhost:8001")
DIAGNOSIS_URL       = os.getenv("DIAGNOSIS_URL",         "http://localhost:8002")
LAB_ANALYSIS_URL    = os.getenv("LAB_ANALYSIS_URL",      "http://localhost:8003")
DRUG_SAFETY_URL     = os.getenv("DRUG_SAFETY_URL",        "http://localhost:8004")
IMAGING_TRIAGE_URL  = os.getenv("IMAGING_TRIAGE_URL",    "http://localhost:8005")
DIGITAL_TWIN_URL    = os.getenv("DIGITAL_TWIN_URL",       "http://localhost:8006")
CONSENSUS_URL       = os.getenv("CONSENSUS_URL",          "http://localhost:8007")
EXPLANATION_URL     = os.getenv("EXPLANATION_URL",        "http://localhost:8009")

INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "meditwin-internal")


# ── Safe caller wrapper ───────────────────────────────────────────────────────

async def safe_call(
    name: str,
    url: str,
    payload: dict,
    timeout: float = 30.0,
) -> Optional[dict]:
    """
    Call an agent endpoint safely.
    Returns None on any failure — caller must handle None gracefully.
    Logs all errors to stdout (orchestrator error_log handled by node functions).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {INTERNAL_TOKEN}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        logger.warning(f"[{name}] Timeout after {timeout}s — returning None")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"[{name}] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"[{name}] Unexpected error: {e}")
        return None


# ── Per-agent call functions ──────────────────────────────────────────────────

async def call_patient_context(
    patient_id: str,
    fhir_base_url: str,
    sharp_token: str,
) -> Optional[dict]:
    result = await safe_call(
        "PatientContext",
        f"{PATIENT_CONTEXT_URL}/fetch",
        {
            "patient_id": patient_id,
            "fhir_base_url": fhir_base_url,
            "sharp_token": sharp_token or None,
        },
        timeout=20.0,
    )
    if result:
        return result.get("patient_state")  # unwrap the wrapper
    return None


async def call_diagnosis(
    patient_state: dict,
    chief_complaint: str,
) -> Optional[dict]:
    return await safe_call(
        "Diagnosis",
        f"{DIAGNOSIS_URL}/diagnose",
        {
            "patient_state": patient_state,
            "chief_complaint": chief_complaint,
            "include_fhir_resources": True,
        },
        timeout=60.0,   # LLM + RAG — allow more time
    )


async def call_lab_analysis(
    patient_state: dict,
    diagnosis_output: Optional[dict],
) -> Optional[dict]:
    return await safe_call(
        "LabAnalysis",
        f"{LAB_ANALYSIS_URL}/analyze-labs",
        {
            "patient_state": patient_state,
            "diagnosis_agent_output": diagnosis_output,
        },
        timeout=45.0,   # LLM interpretation layer
    )


async def call_drug_safety(
    patient_state: dict,
    diagnosis_output: Optional[dict],
) -> Optional[dict]:
    # Build proposed medications from diagnosis recommended next steps
    # + use patient's current medications as context
    current_meds = [m.get("drug", "") for m in patient_state.get("medications", [])]
    allergies = patient_state.get("allergies", [])
    conditions = patient_state.get("active_conditions", [])

    # Propose first-line antibiotics based on diagnosis if available
    proposed = []
    if diagnosis_output:
        top_code = diagnosis_output.get("top_icd10_code", "")
        # CAP diagnosis → propose standard first-line treatment for safety check
        if top_code.startswith("J1"):
            proposed = ["Amoxicillin 1g", "Azithromycin 500mg"]
        elif top_code.startswith("J4"):  # COPD
            proposed = ["Azithromycin 500mg", "Prednisone 40mg"]

    if not proposed:
        proposed = ["Azithromycin 500mg"]   # safe default for respiratory

    return await safe_call(
        "DrugSafety",
        f"{DRUG_SAFETY_URL}/check-safety",
        {
            "proposed_medications": proposed,
            "current_medications": current_meds,
            "patient_allergies": allergies,
            "active_conditions": conditions,
            "patient_id": patient_state.get("patient_id", "unknown"),
        },
        timeout=30.0,
    )


async def call_imaging(
    patient_state: dict,
    image_data: str,
) -> Optional[dict]:
    return await safe_call(
        "ImagingTriage",
        f"{IMAGING_TRIAGE_URL}/analyze-xray",
        {
            "patient_id": patient_state.get("patient_id", "unknown"),
            "image_data": {
                "format": "base64",
                "content_type": "image/jpeg",
                "data": image_data,
            },
            "patient_context": {
                "age": patient_state.get("demographics", {}).get("age", 40),
                "gender": patient_state.get("demographics", {}).get("gender", "unknown"),
            },
        },
        timeout=30.0,
    )


async def call_digital_twin(
    patient_state: dict,
    diagnosis_output: Optional[dict],
    drug_safety_output: Optional[dict],
) -> Optional[dict]:
    # Build treatment options from drug safety approved medications
    treatment_options = []
    if drug_safety_output and drug_safety_output.get("approved_medications"):
        approved = drug_safety_output.get("approved_medications", [])
        # Option A: first approved drug (outpatient approach)
        if approved:
            treatment_options.append({
                "option_id": "A",
                "label": f"{approved[0]} — outpatient",
                "drugs": [approved[0]],
                "interventions": ["O2 supplementation"],
            })
        # Option B: all approved drugs + hospitalization
        if len(approved) >= 1:
            treatment_options.append({
                "option_id": "B",
                "label": f"IV therapy + hospitalization",
                "drugs": approved[:2],
                "interventions": ["Hospitalization", "IV fluids", "Continuous monitoring"],
            })
    else:
        # Fallback options
        treatment_options = [
            {"option_id": "A", "label": "Azithromycin outpatient",
             "drugs": ["Azithromycin 500mg"], "interventions": ["O2 supplementation"]},
            {"option_id": "B", "label": "IV antibiotics + hospitalization",
             "drugs": ["Ceftriaxone 1g IV", "Azithromycin 500mg"],
             "interventions": ["Hospitalization", "IV fluids"]},
        ]

    diagnosis_str = (
        f"{diagnosis_output.get('top_diagnosis', 'Unknown')} "
        f"({diagnosis_output.get('top_icd10_code', '')})"
        if diagnosis_output else "Unknown"
    )

    return await safe_call(
        "DigitalTwin",
        f"{DIGITAL_TWIN_URL}/simulate",
        {
            "patient_state": patient_state,
            "diagnosis": diagnosis_str,
            "treatment_options": treatment_options,
        },
        timeout=45.0,
    )


async def call_consensus(
    patient_state: dict,
    diagnosis_output: Optional[dict],
    lab_output: Optional[dict],
    imaging_output: Optional[dict],
    drug_safety_output: Optional[dict],
) -> Optional[dict]:
    return await safe_call(
        "Consensus",
        f"{CONSENSUS_URL}/consensus",
        {
            "diagnosis_output":   diagnosis_output,
            "lab_output":         lab_output,
            "imaging_output":     imaging_output,
            "drug_safety_output": drug_safety_output,
            "patient_state":      patient_state,
        },
        timeout=45.0,
    )


async def call_explanation(
    patient_state: dict,
    consensus_output: dict,
    diagnosis_output: Optional[dict],
    lab_output: Optional[dict],
    imaging_output: Optional[dict],
    drug_safety_output: Optional[dict],
    digital_twin_output: Optional[dict],
    chief_complaint: str,
) -> Optional[dict]:
    return await safe_call(
        "Explanation",
        f"{EXPLANATION_URL}/explain",
        {
            "patient_state":      patient_state,
            "consensus_output":   consensus_output,
            "diagnosis_output":   diagnosis_output,
            "lab_output":         lab_output,
            "imaging_output":     imaging_output,
            "drug_safety_output": drug_safety_output,
            "digital_twin_output":digital_twin_output,
            "chief_complaint":    chief_complaint,
        },
        timeout=90.0,   # LLM SOAP + patient explanation — most time-consuming
    )