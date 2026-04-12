from __future__ import annotations
 
import asyncio
import os
import sys
import importlib
from typing import Optional, AsyncIterator, Callable, Awaitable
 
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
 
from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, sse_done,
)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# LAB ANALYSIS  (/stream on port 8003)
# ════════════════════════════════════════════════════════════════════════════════
 
lab_router = APIRouter()
 
from typing import List
class LabStreamRequest(BaseModel):
    patient_state: dict
    diagnosis_agent_output: Optional[dict] = None
    previous_lab_results: Optional[List[dict]] = None
 
 
async def _lab_stream(
    patient_state: dict,
    diagnosis_output: Optional[dict],
    previous_labs: Optional[list],
) -> AsyncIterator[str]:
    node = "lab_analysis"
    timer = Timer()
 
    yield evt_status(node, "Starting lab analysis rules engine...", step=1, total=4)
 
    demographics = patient_state.get("demographics", {})
    age    = demographics.get("age", 40)
    gender = demographics.get("gender", "male")
    labs   = patient_state.get("lab_results", [])
 
    # Step 1 — Rules engine
    yield evt_status(node, "Running deterministic rules engine (CRITICAL flags)...",
                     step=1, total=4)
    from rules_engine import (
        classify_all, detect_patterns, generate_critical_alerts,
        compute_overall_severity, compute_severity_score, check_rapid_changes,
    )
    classified  = await asyncio.to_thread(classify_all, labs, age, gender)
    abnormal    = [r for r in classified if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    critical    = [r for r in classified if r["flag"] == "CRITICAL"]
    severity    = await asyncio.to_thread(compute_overall_severity, classified)
    sev_score   = await asyncio.to_thread(compute_severity_score, classified)
 
    yield evt_progress(node,
                       f"Rules engine: {len(abnormal)} abnormal, {len(critical)} critical",
                       pct=25)
 
    # Step 2 — Patterns
    yield evt_status(node, "Detecting clinical patterns...", step=2, total=4)
    patterns       = await asyncio.to_thread(detect_patterns, classified)
    critical_alerts = await asyncio.to_thread(generate_critical_alerts, classified)
 
    yield evt_progress(node,
                       f"Patterns: {len(patterns)} detected, {len(critical_alerts)} critical alerts",
                       pct=50)
 
    # Step 3 — Trend analysis
    trend_analysis = None
    if previous_labs:
        yield evt_status(node, "Performing trend analysis (delta checks)...",
                         step=3, total=4)
        from main import analyze_trends  # reuse from main module
        trend_analysis = analyze_trends(labs, previous_labs, classified)
        yield evt_progress(node, f"Trend analysis: {len(trend_analysis)} trends", pct=65)
 
    # Step 4 — LLM interpretation
    yield evt_status(node, "Running structured LLM lab interpretation...",
                     step=4, total=4)
    from main import run_llm_interpretation, generate_clinical_decision_support
    llm_interp = await asyncio.to_thread(
        run_llm_interpretation, classified, patterns, diagnosis_output, critical_alerts
    )
    cds = generate_clinical_decision_support(classified, patterns, critical_alerts, llm_interp)
 
    yield evt_progress(node, "LLM interpretation complete", pct=90)
 
    # Build final response dict
    if llm_interp:
        confirms        = llm_interp.confirms_top_diagnosis
        confidence_boost = llm_interp.lab_confidence_boost
        alt_code        = llm_interp.alternative_diagnoses[0].icd10_code if llm_interp.alternative_diagnoses else None
        reasoning       = llm_interp.reasoning
        pattern_interp  = llm_interp.pattern_interpretation
    else:
        bacterial = any("bacterial" in p["pattern"].lower() for p in patterns)
        top_icd10 = (diagnosis_output or {}).get("top_icd10_code", "")
        confirms        = not (bacterial and top_icd10 and not top_icd10.startswith("J"))
        confidence_boost = 0.10 if bacterial else 0.0
        alt_code        = None
        reasoning       = "Rules-only mode"
        pattern_interp  = (
            f"Detected: {', '.join(p['pattern'] for p in patterns)}"
            if patterns else "No patterns"
        )
 
    flagged_results = [
        {"loinc": r["loinc"], "display": r["display"], "value": r["value"],
         "unit": r["unit"], "reference_range": r["reference_range"],
         "flag": r["flag"], "clinical_significance": r["clinical_notes"]}
        for r in abnormal
    ]
 
    result = {
        "lab_summary": {
            "total_results":   len(classified),
            "abnormal_count":  len(abnormal),
            "critical_count":  len(critical),
            "overall_severity": severity,
            "severity_score":  sev_score,
        },
        "flagged_results":     flagged_results,
        "pattern_analysis":    {"identified_patterns": patterns,
                                "pattern_interpretation": pattern_interp},
        "diagnosis_confirmation": {
            "confirms_top_diagnosis": confirms,
            "lab_confidence_boost":   round(confidence_boost, 2),
            "alternative_diagnosis_code": alt_code,
            "reasoning": reasoning,
        },
        "critical_alerts":          critical_alerts,
        "trend_analysis":           [t.model_dump() for t in trend_analysis] if trend_analysis else None,
        "severity_score":           sev_score,
        "clinical_decision_support": cds,
        "llm_interpretation_available": llm_interp is not None,
    }
 
    elapsed = timer.elapsed_ms()
    yield evt_complete(node, result, elapsed_ms=elapsed)
 
 
@lab_router.post("/stream")
async def lab_stream(request: LabStreamRequest):
    async def gen():
        async for chunk in _lab_stream(
            request.patient_state,
            request.diagnosis_agent_output,
            request.previous_lab_results,
        ):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)