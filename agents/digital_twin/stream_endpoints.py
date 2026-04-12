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
# DIGITAL TWIN  (/stream on port 8006)
# ════════════════════════════════════════════════════════════════════════════════
 
twin_router = APIRouter()
 
 
class TwinStreamRequest(BaseModel):
    patient_state:     dict
    diagnosis:         str = "Unknown diagnosis"
    diagnosis_code:    Optional[str] = None
    treatment_options: list = []
 
 
async def _twin_stream(req: TwinStreamRequest) -> AsyncIterator[str]:
    node  = "digital_twin"
    timer = Timer()
 
    # Reuse main.py's fully-built simulation pipeline — just wrap with SSE events
    from main import (
        _models, _models_loaded, _llm, _llm_ready, _tools_ready,
        predict_baseline_risks_with_uncertainty,
        _determine_model_confidence, _generate_simulation_hash,
        sanitize,
    )
    from simulator import (
        simulate_treatment, determine_patient_risk_profile, select_recommended_option,
    )
    from feature_engineering import engineer_features, FEATURE_NAMES, get_feature_attribution
    from clinical_tools import (
        check_drug_guideline_adherence,
        perform_sensitivity_analysis, analyze_cost_effectiveness,
        build_enhanced_llm_narrative, build_enhanced_fhir_care_plan,
    )
    from model import TreatmentOption
 
    if not _models_loaded:
        yield evt_error(node, "XGBoost models not loaded — run train_models.py first",
                        fatal=True)
        return
 
    yield evt_status(node, "Engineering patient features...", step=1, total=5)
    try:
        feature_vector, feature_dict = await asyncio.to_thread(
            engineer_features, req.patient_state
        )
    except Exception as exc:
        yield evt_error(node, f"Feature engineering failed: {exc}", fatal=True)
        return
 
    yield evt_status(node, "Running XGBoost risk models with uncertainty quantification...",
                     step=2, total=5)
    try:
        baseline_risks_ci = await asyncio.to_thread(
            predict_baseline_risks_with_uncertainty, feature_vector
        )
    except Exception as exc:
        yield evt_error(node, f"Risk prediction failed: {exc}", fatal=True)
        return
 
    baseline_risks = {
        "readmission_30d": baseline_risks_ci["readmission_30d"]["point_estimate"],
        "mortality_30d":   baseline_risks_ci["mortality_30d"]["point_estimate"],
        "complication":    baseline_risks_ci["complication"]["point_estimate"],
    }
    risk_profile    = determine_patient_risk_profile(baseline_risks)
    model_confidence = _determine_model_confidence(baseline_risks_ci)
 
    yield evt_progress(node,
                       f"Risk profile: {risk_profile} | "
                       f"Mortality 30d: {baseline_risks['mortality_30d']:.0%}",
                       pct=40)
 
    # Treatment options
    treatment_options = [TreatmentOption(**o) if isinstance(o, dict) else o
                         for o in req.treatment_options]
    if not any(o.option_id == "C" for o in treatment_options):
        treatment_options.append(TreatmentOption(
            option_id="C", label="No treatment (baseline)", drugs=[], interventions=[]
        ))
 
    yield evt_status(node, "Simulating treatment scenarios...", step=3, total=5)
    diagnosis_code = req.diagnosis_code or req.diagnosis.split("(")[-1].strip(")")
 
    scenarios = []
    for opt in treatment_options:
        preds = await asyncio.to_thread(
            simulate_treatment,
            baseline_risks, opt.drugs, opt.interventions,
        )
        ga = None
        if _tools_ready and diagnosis_code and opt.drugs:
            try:
                ga = check_drug_guideline_adherence.invoke({
                    "diagnosis_code": diagnosis_code,
                    "proposed_drug": opt.drugs[0],
                })
            except Exception:
                pass
        key_risks = []
        if preds["mortality_risk_30d"] > 0.10:
            key_risks.append(f"30d mortality: {preds['mortality_risk_30d']:.0%}")
        if preds["readmission_risk_30d"] > 0.20:
            key_risks.append(f"Readmission: {preds['readmission_risk_30d']:.0%}")
        if not key_risks:
            key_risks.append("Low overall risk")
        scenarios.append({
            "option_id": opt.option_id, "label": opt.label,
            "drugs": opt.drugs, "interventions": opt.interventions,
            "predictions": preds, "key_risks": key_risks,
            "guideline_adherence": ga,
        })
 
    scoreable    = [s for s in scenarios if s["option_id"] != "C"]
    recommended_id, rec_confidence = (
        select_recommended_option(scoreable) if scoreable else ("A", 0.70)
    )
 
    yield evt_progress(node,
                       f"Recommended: Option {recommended_id} ({rec_confidence:.0%} confidence)",
                       pct=70)
 
    yield evt_status(node, "Generating clinical narrative (LLM)...", step=4, total=5)
    narrative = await asyncio.to_thread(
        build_enhanced_llm_narrative,
        req.patient_state, req.diagnosis, diagnosis_code,
        scenarios, recommended_id, risk_profile,
        _llm, _llm_ready, None, None,
    )
 
    attribution = await asyncio.to_thread(get_feature_attribution, feature_dict, baseline_risks)
 
    yield evt_status(node, "Building FHIR CarePlan...", step=5, total=5)
    rec_opt = next((o for o in treatment_options if o.option_id == recommended_id),
                   treatment_options[0])
    rec_scen = next((s for s in scenarios if s["option_id"] == recommended_id), scenarios[0])
    fhir_care_plan = await asyncio.to_thread(
        build_enhanced_fhir_care_plan,
        req.patient_state.get("patient_id", "unknown"),
        rec_opt, narrative,
        rec_scen["predictions"].get("recovery_probability_7d", 0.7),
        model_confidence, diagnosis_code, attribution,
    )
 
    result = sanitize({
        "simulation_summary": {
            "patient_risk_profile": risk_profile,
            "baseline_risks": baseline_risks,
            "recommended_option": recommended_id,
            "recommendation_confidence": rec_confidence,
            "model_confidence": model_confidence,
        },
        "scenarios":         scenarios,
        "what_if_narrative": narrative,
        "fhir_care_plan":    fhir_care_plan,
        "feature_attribution": attribution,
        "models_loaded":     True,
        "model_confidence":  model_confidence,
        "mock":              False,
    })
 
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@twin_router.post("/stream")
async def twin_stream(request: TwinStreamRequest):
    async def gen():
        async for chunk in _twin_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
 
