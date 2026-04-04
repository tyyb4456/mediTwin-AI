"""
Agent 3: Lab Analysis Agent
Deterministic rules engine + LLM interpretation layer for FHIR Observation resources.
Port: 8003

Architecture (as per strategy doc):
  Step 1 — Rules engine runs FIRST: deterministic, handles CRITICAL flags
  Step 2 — LLM runs AFTER: adds qualitative pattern interpretation only
  CRITICAL alerts are NEVER suppressed by LLM output.
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv  
load_dotenv()

# Add parent directories to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rules_engine import (
    classify_all,
    detect_patterns,
    generate_critical_alerts,
    compute_overall_severity,
)


# ── LLM Interpretation Prompt ─────────────────────────────────────────────────
LAB_INTERPRETATION_SYSTEM = """You are a clinical pathologist interpreting laboratory results.
You receive a patient's abnormal lab values and a proposed diagnosis, then provide clinical interpretation.

CRITICAL RULES:
1. Return ONLY valid JSON — no preamble, no markdown
2. Do NOT contradict CRITICAL flags — they are deterministic
3. Only comment on patterns visible in the data provided
4. Be concise and clinically precise
"""

LAB_INTERPRETATION_USER = """ABNORMAL LAB RESULTS:
{abnormal_results}

DETECTED PATTERNS:
{patterns}

PROPOSED DIAGNOSIS FROM DIAGNOSIS AGENT:
{proposed_diagnosis} (ICD-10: {proposed_icd10})

Analyze the lab findings and return JSON with this exact structure:
{{
  "pattern_interpretation": "2-3 sentence clinical narrative explaining what the lab pattern means together",
  "confirms_top_diagnosis": true or false,
  "lab_confidence_boost": 0.0 to 0.20 (how much these labs add to diagnostic confidence),
  "alternative_diagnosis_code": "ICD-10 code if labs suggest different diagnosis, else null",
  "alternative_diagnosis_display": "display name if alternative, else null",
  "reasoning": "one sentence explaining why labs confirm or contradict the proposed diagnosis"
}}

Return ONLY JSON.
"""


# ── Request / Response Models ──────────────────────────────────────────────────

class LabAnalysisRequest(BaseModel):
    patient_state: dict           # Full PatientState dict
    diagnosis_agent_output: Optional[dict] = None  # Output from Diagnosis Agent (may be None)


class LabAnalysisResponse(BaseModel):
    lab_summary: dict
    flagged_results: list
    pattern_analysis: dict
    diagnosis_confirmation: dict
    critical_alerts: list
    llm_interpretation_available: bool


# ── Lifespan ──────────────────────────────────────────────────────────────────

llm: Optional[ChatGoogleGenerativeAI] = None
llm_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, llm_ready

    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
        llm_ready = True
        print("✓ Lab Analysis Agent started — LLM interpretation ready")
    except Exception as e:
        print(f"⚠️  LLM init failed: {e} — rules engine only mode")


    yield
    print("✓ Lab Analysis Agent shutdown")


app = FastAPI(
    title="MediTwin Lab Analysis Agent",
    description="Deterministic rules engine + LLM interpretation for lab results",
    version="1.0.0",
    lifespan=lifespan,
)


# ── LLM Interpretation Helper ─────────────────────────────────────────────────

def run_llm_interpretation(
    classified_labs: list[dict],
    patterns: list[dict],
    diagnosis_output: Optional[dict],
) -> Optional[dict]:
    """
    Run LLM interpretation on top of rules engine output.
    Returns None if LLM is unavailable — rules engine output is still complete.
    """
    if not llm_ready or not llm:
        return None

    # Only pass abnormal results to LLM — keep prompt tight
    abnormal = [
        f"{r['display']}: {r['value']} {r['unit']} [{r['flag']}] — {r['clinical_notes']}"
        for r in classified_labs
        if r["flag"] in ("HIGH", "LOW", "CRITICAL")
    ]

    if not abnormal:
        return {
            "pattern_interpretation": "All lab values within normal limits. No abnormal pattern detected.",
            "confirms_top_diagnosis": True,
            "lab_confidence_boost": 0.0,
            "alternative_diagnosis_code": None,
            "alternative_diagnosis_display": None,
            "reasoning": "Normal labs do not contradict the proposed diagnosis.",
        }

    # Proposed diagnosis defaults
    proposed_dx = "Unknown"
    proposed_icd10 = "UNKNOWN"
    if diagnosis_output:
        proposed_dx = diagnosis_output.get("top_diagnosis", "Unknown")
        proposed_icd10 = diagnosis_output.get("top_icd10_code", "UNKNOWN")

    pattern_summary = "\n".join(
        f"- {p['pattern']}: {p['description']}" for p in patterns
    ) or "No specific patterns detected"

    prompt = ChatPromptTemplate.from_messages([
        ("system", LAB_INTERPRETATION_SYSTEM),
        ("human", LAB_INTERPRETATION_USER),
    ])

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "abnormal_results": "\n".join(abnormal),
            "patterns": pattern_summary,
            "proposed_diagnosis": proposed_dx,
            "proposed_icd10": proposed_icd10,
        })
        return result
    except Exception as e:
        print(f"⚠️  LLM interpretation failed: {e}")
        return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/analyze-labs", response_model=LabAnalysisResponse)
async def analyze_labs(request: LabAnalysisRequest) -> LabAnalysisResponse:
    """
    Analyze patient lab results.

    Step 1: Rules engine classifies every LOINC result (deterministic)
    Step 2: Pattern detection identifies clinical syndromes
    Step 3: Critical alert generation (never suppressed)
    Step 4: LLM adds qualitative interpretation (optional, on top of rules)
    Step 5: Diagnosis confirmation signal for Consensus Agent
    """
    patient_state = request.patient_state
    if not patient_state:
        raise HTTPException(status_code=400, detail="patient_state is required")

    demographics = patient_state.get("demographics", {})
    age = demographics.get("age", 40)
    gender = demographics.get("gender", "male")
    lab_results = patient_state.get("lab_results", [])

    # ── Step 1: Rules engine — classify ALL results ────────────────────────────
    classified = classify_all(lab_results, age, gender)

    abnormal = [r for r in classified if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    critical = [r for r in classified if r["flag"] == "CRITICAL"]

    severity = compute_overall_severity(classified)

    lab_summary = {
        "total_results": len(classified),
        "abnormal_count": len(abnormal),
        "critical_count": len(critical),
        "overall_severity": severity,
    }

    # ── Step 2: Pattern detection ──────────────────────────────────────────────
    patterns = detect_patterns(classified)

    # ── Step 3: Critical alerts (deterministic, always included) ───────────────
    critical_alerts = generate_critical_alerts(classified)

    # ── Step 4: LLM interpretation (additive, never overrides Step 3) ─────────
    llm_result = run_llm_interpretation(classified, patterns, request.diagnosis_agent_output)

    # ── Step 5: Build diagnosis_confirmation for Consensus Agent ───────────────
    # This is the critical interface field the Consensus Agent reads
    if llm_result:
        confirms = llm_result.get("confirms_top_diagnosis", True)
        confidence_boost = float(llm_result.get("lab_confidence_boost", 0.0))
        alt_code = llm_result.get("alternative_diagnosis_code")
        alt_display = llm_result.get("alternative_diagnosis_display")
        reasoning = llm_result.get("reasoning", "")
        pattern_interp = llm_result.get("pattern_interpretation", "")
    else:
        # Rules-only fallback: if bacterial pattern detected, confirm pneumonia-range diagnoses
        bacterial_pattern = any(
            "bacterial" in p["pattern"].lower() or "sepsis" in p["pattern"].lower()
            for p in patterns
        )
        top_icd10 = (request.diagnosis_agent_output or {}).get("top_icd10_code", "")
        confirms = True
        if bacterial_pattern and top_icd10 and not top_icd10.startswith("J"):
            confirms = False
        confidence_boost = 0.10 if bacterial_pattern else 0.0
        alt_code = None
        alt_display = None
        reasoning = "Rules-only mode — pattern detection used for confirmation."
        pattern_interp = (
            f"Detected patterns: {', '.join(p['pattern'] for p in patterns)}"
            if patterns else "No specific patterns detected."
        )

    # Compute proposed top ICD-10 from Diagnosis Agent (for confirmation)
    proposed_icd10 = (request.diagnosis_agent_output or {}).get("top_icd10_code", "UNKNOWN")
    proposed_dx = (request.diagnosis_agent_output or {}).get("top_diagnosis", "Unknown")

    diagnosis_confirmation = {
        "proposed_diagnosis": proposed_dx,
        "proposed_icd10": proposed_icd10,
        "confirms_top_diagnosis": confirms,
        "lab_confidence_boost": round(confidence_boost, 2),
        "alternative_diagnosis_code": alt_code,
        "alternative_diagnosis_display": alt_display,
        "reasoning": reasoning,
    }

    pattern_analysis = {
        "identified_patterns": patterns,
        "pattern_interpretation": pattern_interp,
    }

    # Build flagged_results (only abnormal, with clinical significance)
    flagged_results = []
    for r in abnormal:
        flagged_results.append({
            "loinc": r["loinc"],
            "display": r["display"],
            "value": r["value"],
            "unit": r["unit"],
            "reference_range": r["reference_range"],
            "flag": r["flag"],
            "clinical_significance": r["clinical_notes"],
        })

    return LabAnalysisResponse(
        lab_summary=lab_summary,
        flagged_results=flagged_results,
        pattern_analysis=pattern_analysis,
        diagnosis_confirmation=diagnosis_confirmation,
        critical_alerts=critical_alerts,
        llm_interpretation_available=llm_ready,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "lab-analysis",
        "version": "1.0.0",
        "llm_ready": llm_ready,
        "rules_engine": "active",
        "loinc_codes_supported": 16,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)