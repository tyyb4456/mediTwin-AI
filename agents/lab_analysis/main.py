"""
Agent 3: Lab Analysis Agent (Enhanced)
Deterministic rules engine + structured LLM interpretation + clinical decision support
Port: 8003
"""
import os
import sys
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rules_engine import (
    classify_all,
    detect_patterns,
    generate_critical_alerts,
    compute_overall_severity,
    compute_severity_score,
    check_rapid_changes,
)

# All shared logic lives in lab_logic — no duplication, no circular imports
from lab_logic import (
    LabInterpretation,
    TrendAnalysis,
    analyze_trends,
    run_llm_interpretation,
    generate_clinical_decision_support,
)

# stream_endpoint imports lab_logic directly, so this import is now safe
from stream_endpoint import lab_router as stream_router

from db import init as db_init, close as db_close, save_lab_analysis, LabAnalysisRecord


# ── Request / Response Models ──────────────────────────────────────────────────

class LabAnalysisRequest(BaseModel):
    patient_state: dict
    diagnosis_agent_output: Optional[dict] = None
    previous_lab_results: Optional[List[dict]] = None
    request_id: Optional[str] = None   # correlation ID, generated if not provided


class LabAnalysisResponse(BaseModel):
    request_id: str
    lab_summary: dict
    flagged_results: list
    pattern_analysis: dict
    diagnosis_confirmation: dict
    critical_alerts: list
    trend_analysis: Optional[List[TrendAnalysis]] = None
    severity_score: Optional[dict] = None
    clinical_decision_support: Optional[dict] = None
    llm_interpretation_available: bool


# ── LLM singleton — set during lifespan, read by stream_endpoint at call time ─

llm: Optional[ChatGoogleGenerativeAI] = None
llm_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, llm_ready

    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-lite",
            temperature=0.2,
        )
        llm_ready = True
        print("✓ Lab Analysis Agent started — Enhanced LLM interpretation ready")
    except Exception as e:
        print(f"⚠️  LLM init failed: {e} — rules engine only mode")

    await db_init()       # ← DB pool startup

    yield

    await db_close()      # ← DB pool shutdown
    print("✓ Lab Analysis Agent shutdown")


app = FastAPI(
    title="MediTwin Lab Analysis Agent (Enhanced)",
    description="Advanced lab interpretation with structured LLM output and clinical decision support",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(stream_router)

# after the existing imports
from history_router import router as history_router
app.include_router(history_router, prefix="/history", tags=["history"])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze-labs", response_model=LabAnalysisResponse)
async def analyze_labs(request: LabAnalysisRequest) -> LabAnalysisResponse:
    patient_state = request.patient_state
    if not patient_state:
        raise HTTPException(status_code=400, detail="patient_state is required")

    request_id = request.request_id or str(uuid.uuid4())[:8]
    patient_id = patient_state.get("patient_id", "unknown")

    demographics = patient_state.get("demographics", {})
    age    = demographics.get("age", 40)
    gender = demographics.get("gender", "male")
    lab_results = patient_state.get("lab_results", [])

    # ── Step 1: Rules engine ───────────────────────────────────────────────────
    classified  = classify_all(lab_results, age, gender)
    abnormal    = [r for r in classified if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    critical    = [r for r in classified if r["flag"] == "CRITICAL"]
    severity    = compute_overall_severity(classified)
    sev_score   = compute_severity_score(classified)

    lab_summary = {
        "total_results":    len(classified),
        "abnormal_count":   len(abnormal),
        "critical_count":   len(critical),
        "overall_severity": severity,
    }

    # ── Step 2: Pattern detection ──────────────────────────────────────────────
    patterns = detect_patterns(classified)

    # ── Step 3: Critical alerts ────────────────────────────────────────────────
    critical_alerts = generate_critical_alerts(classified)

    # ── Step 4: Trend analysis ─────────────────────────────────────────────────
    trend_analysis = None
    if request.previous_lab_results:
        trend_analysis = analyze_trends(lab_results, request.previous_lab_results, classified)

    # ── Step 5: Structured LLM interpretation ──────────────────────────────────
    llm_interpretation = run_llm_interpretation(
        classified, patterns, request.diagnosis_agent_output, critical_alerts,
        llm=llm, llm_ready=llm_ready,
    )

    # ── Step 6: Clinical decision support ──────────────────────────────────────
    cds = generate_clinical_decision_support(classified, patterns, critical_alerts, llm_interpretation)

    # ── Step 7: Build diagnosis_confirmation ───────────────────────────────────
    proposed_dx    = (request.diagnosis_agent_output or {}).get("top_diagnosis", "Unknown")
    proposed_icd10 = (request.diagnosis_agent_output or {}).get("top_icd10_code", "UNKNOWN")

    if llm_interpretation:
        confirms         = llm_interpretation.confirms_top_diagnosis
        confidence_boost = llm_interpretation.lab_confidence_boost
        alt_code         = llm_interpretation.alternative_diagnoses[0].icd10_code if llm_interpretation.alternative_diagnoses else None
        alt_display      = llm_interpretation.alternative_diagnoses[0].display if llm_interpretation.alternative_diagnoses else None
        reasoning        = llm_interpretation.reasoning
        pattern_interp   = llm_interpretation.pattern_interpretation
    else:
        bacterial    = any("bacterial" in p["pattern"].lower() or "sepsis" in p["pattern"].lower() for p in patterns)
        confirms     = not (bacterial and proposed_icd10 and not proposed_icd10.startswith("J"))
        confidence_boost = 0.10 if bacterial else 0.0
        alt_code     = None
        alt_display  = None
        reasoning    = "Rules-only mode — pattern detection used for confirmation"
        pattern_interp = (
            f"Detected patterns: {', '.join(p['pattern'] for p in patterns)}"
            if patterns else "No specific patterns detected"
        )

    diagnosis_confirmation = {
        "proposed_diagnosis":            proposed_dx,
        "proposed_icd10":                proposed_icd10,
        "confirms_top_diagnosis":        confirms,
        "lab_confidence_boost":          round(confidence_boost, 2),
        "alternative_diagnosis_code":    alt_code,
        "alternative_diagnosis_display": alt_display,
        "reasoning":                     reasoning,
    }

    pattern_analysis = {
        "identified_patterns":    patterns,
        "pattern_interpretation": pattern_interp,
    }

    flagged_results = [
        {
            "loinc": r["loinc"],
            "display": r["display"],
            "value": r["value"],
            "unit": r["unit"],
            "reference_range": r["reference_range"],
            "flag": r["flag"],
            "clinical_significance": r["clinical_notes"],
        }
        for r in abnormal
    ]

    # ── Step 8: Persist to DB (non-fatal) ─────────────────────────────────────
    await save_lab_analysis(LabAnalysisRecord(
        request_id=request_id,
        patient_id=patient_id,
        total_results=len(classified),
        abnormal_count=len(abnormal),
        critical_count=len(critical),
        overall_severity=severity,
        severity_score=sev_score,
        flagged_results=flagged_results,
        identified_patterns=patterns,
        critical_alerts=critical_alerts,
        confirms_top_diagnosis=confirms,
        proposed_diagnosis=proposed_dx,
        proposed_icd10=proposed_icd10,
        lab_confidence_boost=confidence_boost,
        alternative_diagnosis_code=alt_code,
        clinical_decision_support=cds,
        trend_analysis=[t.model_dump() for t in trend_analysis] if trend_analysis else None,
        llm_interpretation_available=llm_ready,
        cache_hit=False,
        source="analyze-labs",
    ))

    return LabAnalysisResponse(
        request_id=request_id,
        lab_summary=lab_summary,
        flagged_results=flagged_results,
        pattern_analysis=pattern_analysis,
        diagnosis_confirmation=diagnosis_confirmation,
        critical_alerts=critical_alerts,
        trend_analysis=trend_analysis,
        severity_score=sev_score,
        clinical_decision_support=cds,
        llm_interpretation_available=llm_ready,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "lab-analysis-enhanced",
        "version": "2.0.0",
        "llm_ready": llm_ready,
        "structured_output": llm_ready,
        "rules_engine": "active",
        "loinc_codes_supported": 16,
        "features": [
            "structured_llm_output",
            "trend_analysis",
            "clinical_decision_support",
            "severity_scoring",
            "delta_checks",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)