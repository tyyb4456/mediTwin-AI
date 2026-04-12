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
# EXPLANATION  (/stream on port 8009)
# ════════════════════════════════════════════════════════════════════════════════
 
explanation_router = APIRouter()
 
 
class ExplanationStreamRequest(BaseModel):
    patient_state:      dict
    consensus_output:   dict
    diagnosis_output:   Optional[dict] = None
    lab_output:         Optional[dict] = None
    imaging_output:     Optional[dict] = None
    drug_safety_output: Optional[dict] = None
    digital_twin_output: Optional[dict] = None
    chief_complaint:    str = "Not specified"
 
 
async def _explanation_stream(req: ExplanationStreamRequest) -> AsyncIterator[str]:
    node  = "explanation"
    timer = Timer()
 
    from soap_generator    import generate_soap_note, MEDICAL_DISCLAIMER
    from patient_writer    import generate_patient_explanation
    from fhir_bundler      import build_fhir_bundle
    from main              import build_risk_attribution, extract_risk_flags
 
    yield evt_status(node, "Generating SOAP note (LLM)...", step=1, total=4)
    try:
        soap = await asyncio.to_thread(
            generate_soap_note,
            req.patient_state, req.consensus_output,
            req.lab_output, req.imaging_output,
            req.drug_safety_output, req.digital_twin_output,
            req.chief_complaint,
        )
    except Exception as exc:
        soap = {
            "subjective": f"Error: {exc}",
            "objective":  "See individual agent outputs",
            "assessment": "Clinical review required" + MEDICAL_DISCLAIMER,
            "plan":       ["Physician review required"],
            "clinical_summary_one_liner": "Error in SOAP generation",
        }
    yield evt_progress(node, "SOAP note generated", pct=25)
 
    yield evt_status(node,
                     "Generating patient explanation (grade-6 reading level gate)...",
                     step=2, total=4)
    try:
        patient_dict, reading_stats = await asyncio.to_thread(
            generate_patient_explanation,
            req.patient_state, req.consensus_output,
            req.drug_safety_output, req.digital_twin_output, req.imaging_output,
        )
        patient_dict["reading_level_check"] = reading_stats
    except Exception as exc:
        patient_dict  = {
            "condition_explanation": "Your doctor will explain your condition.",
            "why_this_happened":     "Please ask your care team.",
            "what_happens_next":     "Your doctor has a plan for your care.",
            "what_to_expect":        ["Your care team will explain the next steps"],
            "important_for_you_to_know": "Please speak with your nurse or doctor.",
            "when_to_call_the_nurse": ["If you feel worse", "If you have questions"],
        }
        reading_stats = {"grade_level": 6.0, "acceptable": True, "error": str(exc)}
 
    yield evt_progress(node,
                       f"Patient explanation ready — reading grade {reading_stats.get('grade_level', '?')}",
                       pct=55)
 
    yield evt_status(node, "Building risk attribution from XGBoost feature importances...",
                     step=3, total=4)
    risk_attribution = build_risk_attribution(req.digital_twin_output, req.consensus_output)
    yield evt_progress(node, "Risk attribution complete", pct=75)
 
    yield evt_status(node, "Assembling FHIR R4 Bundle...", step=4, total=4)
    try:
        fhir_bundle = await asyncio.to_thread(
            build_fhir_bundle,
            req.patient_state.get("patient_id", "unknown"),
            req.diagnosis_output, req.imaging_output,
            req.drug_safety_output, req.digital_twin_output,
        )
    except Exception as exc:
        fhir_bundle = {"resourceType": "Bundle", "type": "collection",
                       "entry": [], "error": str(exc)}
 
    risk_flags = extract_risk_flags(req.patient_state, req.drug_safety_output, req.lab_output)
 
    result = {
        "clinician_output": {
            "soap_note": soap,
            "clinical_summary_one_liner": soap.get("clinical_summary_one_liner", ""),
            "risk_flags": risk_flags,
            "confidence_breakdown": {
                "overall":          req.consensus_output.get("aggregate_confidence", 0.0),
                "consensus_status": req.consensus_output.get("consensus_status", "UNKNOWN"),
                "conflict_count":   req.consensus_output.get("conflict_count", 0),
            },
        },
        "patient_output":         patient_dict,
        "risk_attribution":       risk_attribution,
        "fhir_bundle":            fhir_bundle,
        "reading_level_check":    reading_stats,
        "consensus_status":       req.consensus_output.get("consensus_status", "UNKNOWN"),
        "human_review_required":  req.consensus_output.get("human_review_required", False),
    }
 
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@explanation_router.post("/stream")
async def explanation_stream(request: ExplanationStreamRequest):
    async def gen():
        async for chunk in _explanation_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)