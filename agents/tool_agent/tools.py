"""
MediTwin Tool Agent — Enhanced Tool Definitions
Each downstream specialist agent is exposed as a @tool with improved descriptions
and better conversational output formatting.

ENHANCEMENTS:
- More natural tool descriptions (less mechanical, more helpful)
- Better structured outputs that are easier for LLM to synthesize
- Clearer error messages
- Progress indicators during long-running operations
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


# ── Patient state normalization (unchanged but with better logging) ────────────

def _unwrap_patient_state(raw) -> dict:
    """
    Normalize whatever the LLM passes as patient_state into a clean dict.
    Enhanced with better error messages.
    """
    if isinstance(raw, str):
        try:
            return _unwrap_patient_state(json.loads(raw))
        except json.JSONDecodeError:
            logger.error(f"patient_state is an unparseable string: {raw[:100]}")
            return {}

    if not isinstance(raw, dict):
        logger.error(f"patient_state is unexpected type: {type(raw)}")
        return {}

    if "patient_id" in raw:
        return raw

    if len(raw) == 1:
        inner = next(iter(raw.values()))
        return _unwrap_patient_state(inner)

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


# ── Async HTTP helper with better error messages ──────────────────────────────

async def _post(name: str, url: str, payload: dict, timeout: float = 45.0) -> dict:
    """
    HTTP POST with enhanced error handling and user-friendly messages.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=HEADERS)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        logger.warning(f"[{name}] Timeout after {timeout}s")
        return {
            "error": f"{name} timed out",
            "user_message": f"The {name.lower()} analysis is taking longer than expected. This might indicate a complex case or system load.",
            "suggestion": "Try again, or ask me to focus on a specific aspect of the analysis."
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"[{name}] HTTP {e.response.status_code}")
        return {
            "error": f"{name} returned HTTP {e.response.status_code}",
            "user_message": f"There was an issue connecting to the {name.lower()} service.",
            "detail": e.response.text[:300]
        }
    except Exception as e:
        logger.error(f"[{name}] Unexpected error: {e}")
        return {
            "error": f"{name} failed",
            "user_message": f"An unexpected error occurred during {name.lower()}.",
            "technical_detail": str(e)
        }
 
 
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
# ENHANCED TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def fetch_patient_context(
    patient_id: str,
    fhir_base_url: str = "https://hapi.fhir.org/baseR4",
    sharp_token: Optional[str] = None,
) -> dict:
    """
    Retrieve the complete clinical profile for a patient from their electronic health record.
 
    This is your starting point for any patient-specific analysis. It gathers:
    - Demographics (age, gender, date of birth)
    - Active medical conditions (with ICD-10 codes)
    - Current medications (with dosing)
    - Documented allergies and reactions
    - Recent laboratory results (last 20 tests)
    - Diagnostic reports and imaging availability
    - Recent clinical encounters
    
    The data is pulled directly from FHIR-compliant health records and cached for 10 minutes
    to improve response time for follow-up queries about the same patient.
 
    Use this tool first whenever analyzing a specific patient. All other clinical tools
    require the patient_state dict returned by this function.
 
    Args:
        patient_id: FHIR Patient resource ID (e.g., "example", "patient-123")
        fhir_base_url: FHIR R4 server URL (default: HAPI public sandbox)
        sharp_token: Optional authentication token for protected FHIR servers
    
    Returns:
        A comprehensive patient_state dict containing all clinical data, or an error dict
        if the patient cannot be found or the FHIR server is unreachable.
    """
    _emit({
        "type": "tool_start",
        "tool": "fetch_patient_context",
        "message": f"📋 Retrieving clinical records for patient {patient_id}...",
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
            "message": f"❌ Could not retrieve patient {patient_id}: {result.get('user_message', result['error'])}",
        })
        return result
 
    ps = result.get("patient_state", {})
    demo = ps.get("demographics", {})
    name = demo.get("name", "Unknown")
    conditions = len(ps.get("active_conditions", []))
    meds = len(ps.get("medications", []))
    labs = len(ps.get("lab_results", []))
    
    # Create a human-readable summary
    summary = f"Retrieved records for {name}"
    if conditions > 0:
        summary += f" • {conditions} active condition{'s' if conditions != 1 else ''}"
    if meds > 0:
        summary += f" • {meds} medication{'s' if meds != 1 else ''}"
    if labs > 0:
        summary += f" • {labs} recent lab result{'s' if labs != 1 else ''}"
    
    _emit({
        "type": "tool_complete",
        "tool": "fetch_patient_context",
        "message": f"✓ {summary}",
        "data": {
            "patient_name": name,
            "patient_id": patient_id,
            "cache_hit": result.get("cache_hit", False),
            "conditions_count": conditions,
            "medications_count": meds,
            "labs_count": labs,
        },
    })
 
    return ps


@tool
def run_diagnosis(patient_state: dict, chief_complaint: str) -> dict:
    """
    Generate a ranked differential diagnosis using clinical AI reasoning and medical knowledge.
 
    This tool analyzes the patient's complete clinical picture - their age, gender, current
    conditions, medications, allergies, and lab results - in the context of their presenting
    complaint to produce a clinically-grounded differential diagnosis.
    
    It uses:
    - RAG (Retrieval-Augmented Generation) over medical literature and guidelines
    - Clinical reasoning chains that mirror how physicians think through cases
    - Evidence-based diagnostic criteria (e.g., CURB-65 for pneumonia, Centor for pharyngitis)
    - Confidence scoring based on supporting and contradicting evidence
    
    Returns not just diagnoses, but also:
    - Clinical reasoning for each differential
    - Recommended next steps (labs, imaging, treatments)
    - Safety flags (sepsis risk, allergy alerts, isolation requirements)
 
    When to use:
    - "What's the diagnosis?"
    - "What could be causing these symptoms?"
    - "I need a differential for this patient"
    - Any diagnostic reasoning query
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context (pass as-is, don't modify)
        chief_complaint: The patient's primary symptom or reason for visit (be specific)
    
    Returns:
        Differential diagnosis with ICD-10 codes, confidence scores, clinical reasoning,
        supporting/contradicting evidence, recommended next steps, and safety flags.
    """
    _emit({
        "type": "tool_start",
        "tool": "run_diagnosis",
        "message": f"🔬 Analyzing clinical presentation: {chief_complaint[:60]}...",
    })
 
    patient_state = _unwrap_patient_state(patient_state)
    if not patient_state.get("patient_id"):
        error_msg = "Patient data is incomplete - please run fetch_patient_context first"
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": f"❌ {error_msg}"})
        return {"error": error_msg, "user_message": "I need the patient's clinical data before I can generate a diagnosis."}

    result = _run(_post(
        "Diagnosis",
        f"{DIAGNOSIS_URL}/diagnose",
        {"patient_state": patient_state, "chief_complaint": chief_complaint, "include_fhir_resources": True},
        timeout=60.0,
    ))
 
    if "error" in result:
        _emit({"type": "tool_error", "tool": "run_diagnosis", "message": f"❌ {result.get('user_message', 'Diagnosis analysis failed')}"})
    else:
        top_dx = result.get("top_diagnosis", "Unknown")
        top_code = result.get("top_icd10_code", "")
        confidence = result.get("confidence_level", "")
        n_diff = len(result.get("differential_diagnosis", []))
        
        # Extract important clinical flags
        flags = []
        if result.get("penicillin_allergy_flagged"):
            flags.append("⚠️ Penicillin allergy")
        if result.get("high_suspicion_sepsis"):
            flags.append("🚨 Sepsis risk")
        if result.get("requires_isolation"):
            flags.append("⚠️ Isolation required")
        
        flag_str = " | " + " | ".join(flags) if flags else ""
        
        _emit({
            "type": "tool_complete",
            "tool": "run_diagnosis",
            "message": f"✓ Top diagnosis: {top_dx} ({top_code}) — {confidence} confidence | {n_diff} differentials considered{flag_str}",
            "data": {
                "top_diagnosis": top_dx,
                "top_icd10_code": top_code,
                "confidence_level": confidence,
                "differential_count": n_diff,
                "safety_flags": flags,
            },
        })
 
    return result


@tool
def analyze_labs(patient_state: dict, diagnosis: Optional[dict] = None) -> dict:
    """
    Perform in-depth clinical interpretation of laboratory results.
 
    This goes beyond simple reference range checking to provide:
    - Clinical significance of abnormal values in context
    - Pattern recognition (e.g., SIRS criteria, metabolic acidosis, liver dysfunction)
    - Trend analysis if prior labs are available
    - Diagnosis confirmation or alternative suggestions
    - Severity scoring (0-100 scale with organ system breakdown)
    - Prioritized clinical actions based on critical findings
    
    The tool uses both deterministic rules (for critical values and clinical criteria)
    and AI interpretation to explain what the lab results mean for this specific patient.
 
    When to use:
    - "What do the labs show?"
    - "Are the labs consistent with [diagnosis]?"
    - "Explain the abnormal lab results"
    - "What's the severity based on labs?"
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        diagnosis: Optional - output from run_diagnosis for richer contextualization
    
    Returns:
        Lab summary with flagged results, pattern analysis, diagnosis confirmation,
        severity scoring, trend analysis, and prioritized clinical decision support.
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_labs",
        "message": "🧪 Analyzing laboratory results...",
    })
 
    patient_state = _unwrap_patient_state(patient_state)
    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": f"❌ {error_msg}"})
        return {"error": error_msg}
 
    n_labs = len(patient_state.get("lab_results", []))
    _emit({
        "type": "tool_progress",
        "tool": "analyze_labs",
        "message": f"Processing {n_labs} lab results and checking clinical patterns...",
    })
 
    result = _run(_post(
        "LabAnalysis",
        f"{LAB_ANALYSIS_URL}/analyze-labs",
        {"patient_state": patient_state, "diagnosis_agent_output": diagnosis},
        timeout=45.0,
    ))
 
    if "error" in result:
        _emit({"type": "tool_error", "tool": "analyze_labs", "message": f"❌ {result.get('user_message', 'Lab analysis failed')}"})
    else:
        summary = result.get("lab_summary", {})
        critical = summary.get("critical_count", 0)
        abnormal = summary.get("abnormal_count", 0)
        severity = summary.get("overall_severity", "NORMAL")
        n_alerts = len(result.get("critical_alerts", []))
        
        severity_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "⚡", "LOW": "ℹ️", "NORMAL": "✓"}.get(severity, "")
        
        msg = f"{severity_emoji} Lab severity: {severity}"
        if critical > 0:
            msg += f" | {critical} critical value{'s' if critical != 1 else ''}"
        if abnormal > 0:
            msg += f" | {abnormal} abnormal"
        if n_alerts > 0:
            msg += f" | {n_alerts} clinical alert{'s' if n_alerts != 1 else ''}"
        
        _emit({
            "type": "tool_complete",
            "tool": "analyze_labs",
            "message": msg,
            "data": {
                "critical_count": critical,
                "abnormal_count": abnormal,
                "severity": severity,
                "alerts_count": n_alerts,
            },
        })
 
    return result


@tool
def check_drug_safety(patient_state: dict, proposed_medications: List[str]) -> dict:
    """
    Comprehensive medication safety check against patient-specific factors.
 
    This is a critical safety tool that checks proposed medications against:
    - **Allergies**: Cross-reactivity patterns (e.g., penicillin → cephalosporins)
    - **Drug-drug interactions**: Current medications × proposed medications
    - **Contraindications**: Active conditions that make drugs unsafe
    - **FDA warnings**: Black box warnings and serious safety alerts
    - **Lab-driven concerns**: Renal/hepatic impairment affecting drug metabolism
    
    The tool provides:
    - SAFE/UNSAFE status with clinical explanations
    - Specific reasons for each flagged medication
    - Safer alternative suggestions when drugs are contraindicated
    - Approved medications from the proposed list
    - Pharmacokinetic adjustments when needed
 
    When to use:
    - "Is [drug] safe for this patient?"
    - "Check these antibiotics for safety"
    - "Any drug interactions I should know about?"
    - Before prescribing any medication
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        proposed_medications: List of medication names (can include doses, e.g., "Amoxicillin 500mg")
    
    Returns:
        Safety status, flagged medications, critical interactions, contraindications,
        FDA warnings, approved safe medications, and alternative suggestions.
    """
    med_list = ", ".join(proposed_medications[:3])
    if len(proposed_medications) > 3:
        med_list += f" and {len(proposed_medications) - 3} more"
    
    _emit({
        "type": "tool_start",
        "tool": "check_drug_safety",
        "message": f"💊 Checking safety for: {med_list}...",
    })

    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": f"❌ {error_msg}"})
        return {"error": error_msg}

    state = _unwrap_patient_state(patient_state)

    current_meds = [m.get("drug", "") for m in state.get("medications", [])]
    allergies    = state.get("allergies", [])
    conditions   = state.get("active_conditions", [])

    _emit({
        "type": "tool_progress",
        "tool": "check_drug_safety",
        "message": f"Checking allergy cross-reactivity, drug interactions with {len(current_meds)} current meds, and contraindications...",
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
            "patient_state":        state,  # Full state for lab context
        },
        timeout=30.0,
    ))

    if "error" in result:
        _emit({"type": "tool_error", "tool": "check_drug_safety", "message": f"❌ {result.get('user_message', 'Safety check failed')}"})
    else:
        status = result.get("safety_status", "UNKNOWN")
        flagged = result.get("flagged_medications", [])
        approved = result.get("approved_medications", [])
        interactions = len(result.get("critical_interactions", []))
        
        status_emoji = {"SAFE": "✅", "UNSAFE": "🚫", "CAUTION": "⚠️"}.get(status, "")
        
        msg = f"{status_emoji} Safety status: {status}"
        if approved:
            msg += f" | {len(approved)} medication{'s' if len(approved) != 1 else ''} cleared"
        if flagged:
            msg += f" | {len(flagged)} flagged: {', '.join(flagged[:2])}"
        if interactions > 0:
            msg += f" | {interactions} interaction{'s' if interactions != 1 else ''} detected"
        
        _emit({
            "type": "tool_complete",
            "tool": "check_drug_safety",
            "message": msg,
            "data": {
                "safety_status": status,
                "flagged": flagged,
                "approved": approved,
                "interactions_count": interactions,
            },
        })

    return result


@tool
def analyze_chest_xray(patient_state: dict, image_data_base64: str) -> dict:
    """
    AI-powered chest X-ray analysis using a trained convolutional neural network.
 
    This tool uses EfficientNetB0 (AUC 0.981, Precision 0.976) to analyze chest radiographs
    for signs of pneumonia and other consolidative processes. The analysis includes:
    - **CNN prediction**: PNEUMONIA vs NORMAL with confidence score
    - **Severity grading**: SEVERE → MODERATE → MILD → NORMAL
    - **Triage priority**: 1 (IMMEDIATE) to 4 (ROUTINE) based on urgency
    - **Clinical interpretation**: LLM-generated radiologist-style reading
    - **Imaging findings**: Consolidation patterns, affected areas, bilateral involvement
    - **Recommended actions**: Next steps based on findings
    
    IMPORTANT: Only call this when:
    1. The user explicitly mentions a chest X-ray or imaging
    2. You have actual base64-encoded image data
    
    Do NOT call this tool speculatively or without image data.
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        image_data_base64: Base64-encoded JPEG/PNG chest X-ray image
    
    Returns:
        Model prediction, severity assessment, imaging findings, clinical interpretation,
        diagnosis confirmation, and recommended actions. Includes FHIR DiagnosticReport.
    """
    _emit({
        "type": "tool_start",
        "tool": "analyze_chest_xray",
        "message": "📷 Analyzing chest X-ray with AI (EfficientNetB0 CNN)...",
    })

    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": f"❌ {error_msg}"})
        return {"error": error_msg}

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
        _emit({"type": "tool_error", "tool": "analyze_chest_xray", "message": f"❌ {result.get('user_message', 'Imaging analysis failed')}"})
    else:
        prediction = result.get("model_output", {}).get("prediction", "UNKNOWN")
        confidence = result.get("model_output", {}).get("confidence", 0)
        triage = result.get("severity_assessment", {}).get("triage_label", "UNKNOWN")
        mock = result.get("mock", False)
        
        emoji = {"PNEUMONIA": "🫁", "NORMAL": "✓"}.get(prediction, "📊")
        mock_note = " [Note: Using mock data - model not loaded]" if mock else ""
        
        msg = f"{emoji} Imaging: {prediction} ({confidence:.0%} confidence) | Triage: {triage}{mock_note}"
        
        _emit({
            "type": "tool_complete",
            "tool": "analyze_chest_xray",
            "message": msg,
            "data": {
                "prediction": prediction,
                "confidence": confidence,
                "triage": triage,
                "is_mock": mock,
            },
        })

    return result


@tool
def simulate_treatment_outcomes(
    patient_state: dict,
    diagnosis: dict,
    treatment_options: List[dict],
) -> dict:
    """
    Predict patient outcomes under different treatment scenarios using a Digital Twin simulation.
 
    This advanced tool uses trained XGBoost models to simulate "what if" scenarios:
    - What happens if we give Treatment A vs Treatment B vs no treatment?
    - Which option minimizes mortality and readmission risk?
    - What's the expected recovery timeline?
    - Are there cost-effectiveness trade-offs?
    
    The simulation accounts for:
    - Patient-specific risk factors (age, comorbidities, organ function)
    - Treatment effects (drug combinations, synergies, adherence)
    - Temporal progression (7-day, 30-day, 90-day, 1-year horizons)
    - Uncertainty quantification (credible intervals)
    - Cost-effectiveness (cost per QALY)
    
    Returns guidance on which treatment option is recommended and why.
 
    When to use:
    - "What are the treatment options?"
    - "Compare Treatment A vs Treatment B"
    - "What's the prognosis?"
    - "Simulate outcomes for this patient"
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        diagnosis: Output from run_diagnosis (required for context)
        treatment_options: List of option dicts, each with:
            - option_id (str): "A", "B", "C", etc.
            - label (str): "IV Ceftriaxone + Azithromycin", etc.
            - drugs (List[str]): ["Ceftriaxone 1g IV", "Azithromycin 500mg PO"]
            - interventions (List[str]): ["Hospitalization", "O2 therapy"]
    
    Returns:
        Patient risk profile, baseline risks, scenario comparisons with predictions,
        recommended option, what-if narrative, FHIR CarePlan, and cost-effectiveness analysis.
    """
    n_options = len(treatment_options)
    _emit({
        "type": "tool_start",
        "tool": "simulate_treatment_outcomes",
        "message": f"🤖 Running Digital Twin simulation for {n_options} treatment option{'s' if n_options != 1 else ''}...",
    })

    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ {error_msg}"})
        return {"error": error_msg}
    
    if not isinstance(diagnosis, dict):
        error_msg = "Diagnosis data required for simulation context"
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ {error_msg}"})
        return {"error": error_msg}

    state = _unwrap_patient_state(patient_state)
    
    option_labels = [o.get("label", o.get("option_id", "?")) for o in treatment_options]
    _emit({
        "type": "tool_progress",
        "tool": "simulate_treatment_outcomes",
        "message": f"Simulating outcomes for: {', '.join(option_labels)}...",
    })

    diagnosis_str = f"{diagnosis.get('top_diagnosis', 'Unknown')} ({diagnosis.get('top_icd10_code', '')})"

    result = _run(_post(
        "DigitalTwin",
        f"{DIGITAL_TWIN_URL}/simulate",
        {"patient_state": state, "diagnosis": diagnosis_str, "treatment_options": treatment_options},
        timeout=45.0,
    ))

    if "error" in result:
        _emit({"type": "tool_error", "tool": "simulate_treatment_outcomes", "message": f"❌ {result.get('user_message', 'Simulation failed')}"})
    else:
        summary = result.get("simulation_summary", {})
        risk_profile = summary.get("patient_risk_profile", "UNKNOWN")
        recommended = summary.get("recommended_option", "?")
        rec_confidence = summary.get("recommendation_confidence", 0)
        
        risk_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "✓"}.get(risk_profile, "")
        
        msg = f"✅ Simulation complete | Risk profile: {risk_emoji} {risk_profile} | "
        msg += f"Recommended: Option {recommended} ({rec_confidence:.0%} confidence)"
        
        _emit({
            "type": "tool_complete",
            "tool": "simulate_treatment_outcomes",
            "message": msg,
            "data": {
                "risk_profile": risk_profile,
                "recommended_option": recommended,
                "confidence": rec_confidence,
            },
        })

    return result


@tool
def run_consensus(
    patient_state: dict,
    diagnosis: dict,
    lab_output: Optional[dict] = None,
    imaging_output: Optional[dict] = None,
    drug_safety_output: Optional[dict] = None,
) -> dict:
    """
    Cross-validate outputs from multiple specialist agents and resolve conflicts.
 
    When different AI agents analyze the same patient, they might occasionally disagree
    (e.g., imaging says normal but clinical suspicion is high). This tool:
    - Detects conflicts between agent outputs
    - Attempts resolution using medical knowledge (tiebreaker RAG)
    - Escalates to human review when conflicts are severe or unresolvable
    - Produces a final consensus diagnosis with aggregate confidence
    
    Conflict types detected:
    - Diagnosis ↔ Lab mismatch
    - Imaging ↔ Clinical dissociation
    - Drug safety contraindications
    
    Use this after gathering outputs from multiple tools to get a unified assessment.
 
    When to use:
    - After running diagnosis + labs + imaging
    - Before generating final clinical report
    - When you need a validated consensus
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        diagnosis: Output from run_diagnosis (required)
        lab_output: Optional output from analyze_labs
        imaging_output: Optional output from analyze_chest_xray
        drug_safety_output: Optional output from check_drug_safety
    
    Returns:
        Consensus status (FULL_CONSENSUS / CONFLICT_RESOLVED / ESCALATION_REQUIRED),
        final diagnosis, aggregate confidence, conflict details, human review flag.
    """
    sources = ["diagnosis"]
    if lab_output: sources.append("labs")
    if imaging_output: sources.append("imaging")
    if drug_safety_output: sources.append("drug safety")

    _emit({
        "type": "tool_start",
        "tool": "run_consensus",
        "message": f"⚖️ Cross-validating outputs from {len(sources)} specialist agent{'s' if len(sources) != 1 else ''}: {', '.join(sources)}...",
    })

    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ {error_msg}"})
        return {"error": error_msg}
    
    if not isinstance(diagnosis, dict):
        error_msg = "Diagnosis data required for consensus"
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ {error_msg}"})
        return {"error": error_msg}

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
        _emit({"type": "tool_error", "tool": "run_consensus", "message": f"❌ {result.get('user_message', 'Consensus failed')}"})
    else:
        status = result.get("consensus_status", "UNKNOWN")
        confidence = result.get("aggregate_confidence", 0)
        conflicts = result.get("conflict_count", 0)
        human_review = result.get("human_review_required", False)
        
        status_emoji = {
            "FULL_CONSENSUS": "✅",
            "CONFLICT_RESOLVED": "⚠️",
            "ESCALATION_REQUIRED": "🚨"
        }.get(status, "")
        
        msg = f"{status_emoji} Consensus: {status} | Confidence: {confidence:.0%}"
        if conflicts > 0:
            msg += f" | {conflicts} conflict{'s' if conflicts != 1 else ''} detected"
        if human_review:
            msg += " | ⚠️ HUMAN REVIEW REQUIRED"
        
        _emit({
            "type": "tool_complete",
            "tool": "run_consensus",
            "message": msg,
            "data": {
                "consensus_status": status,
                "confidence": confidence,
                "conflicts": conflicts,
                "human_review_required": human_review,
            },
        })

    return result


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
    Generate the complete clinical documentation package for the patient.
 
    This is typically the final step in a full workup. It synthesizes everything into:
    
    1. **SOAP Note** (for clinicians):
       - Subjective: Patient history and presenting complaint
       - Objective: Vital signs, labs, imaging, medications
       - Assessment: Diagnosis with confidence, risk profile, drug safety status
       - Plan: Ordered list of clinical actions
    
    2. **Patient Explanation** (plain language, grade 6-8 reading level):
       - What's wrong and why it happened
       - What treatment will be done
       - What to expect during treatment
       - When to call the nurse
    
    3. **Risk Attribution** (SHAP-style breakdown):
       - Top factors contributing to readmission/complication risk
       - Helps understand what drives the patient's risk score
    
    4. **FHIR R4 Bundle**:
       - Condition resources (diagnoses)
       - DiagnosticReport (imaging)
       - MedicationRequest (prescriptions)
       - CarePlan (treatment plan)
       - Ready for EHR submission
 
    When to use:
    - "Generate a SOAP note"
    - "Create a clinical summary"
    - "I need the full documentation"
    - At the end of a complete workup
 
    Args:
        patient_state: Complete clinical data from fetch_patient_context
        consensus: Output from run_consensus (required)
        chief_complaint: Patient's primary complaint
        diagnosis: Optional output from run_diagnosis
        lab_output: Optional output from analyze_labs
        imaging_output: Optional output from analyze_chest_xray
        drug_safety_output: Optional output from check_drug_safety
        digital_twin_output: Optional output from simulate_treatment_outcomes
    
    Returns:
        SOAP note, patient explanation, risk attribution, FHIR bundle,
        reading level check, consensus status, and human review flag.
    """
    _emit({
        "type": "tool_start",
        "tool": "generate_clinical_report",
        "message": "📋 Generating clinical documentation (SOAP note, patient explanation, FHIR bundle)...",
    })

    if not isinstance(patient_state, dict):
        error_msg = "Invalid patient data format"
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ {error_msg}"})
        return {"error": error_msg}
    
    if not isinstance(consensus, dict):
        error_msg = "Consensus data required for final report"
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ {error_msg}"})
        return {"error": error_msg}

    patient_state = _unwrap_patient_state(patient_state)

    _emit({
        "type": "tool_progress",
        "tool": "generate_clinical_report",
        "message": "Writing SOAP note and patient explanation (using LLM)...",
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
        _emit({"type": "tool_error", "tool": "generate_clinical_report", "message": f"❌ {result.get('user_message', 'Report generation failed')}"})
    else:
        rl = result.get("reading_level_check", {})
        grade = rl.get("grade_level", "?")
        n_fhir = result.get("fhir_bundle", {}).get("_entry_count", 0)
        consensus_status = result.get("consensus_status", "UNKNOWN")
        
        msg = f"✅ Documentation complete | SOAP note generated | Patient explanation (grade {grade} reading level) | "
        msg += f"{n_fhir} FHIR resource{'s' if n_fhir != 1 else ''} | Status: {consensus_status}"
        
        _emit({
            "type": "tool_complete",
            "tool": "generate_clinical_report",
            "message": msg,
            "data": {
                "reading_grade": grade,
                "fhir_resources": n_fhir,
                "consensus_status": consensus_status,
            },
        })

    return result


# ── Tool registry ─────────────────────────────────────────────────────────────

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