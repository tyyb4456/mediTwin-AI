"""
agents/digital_twin/stream_endpoints.py (ENHANCED VERSION)
------------------------------------------------------------
REPLACE the original stream_endpoints.py with this enhanced version.

Key Enhancements:
1. LLM narrative streaming with token-by-token output
2. Database persistence integration
3. 7-step progress tracking (vs. original 5 steps)
4. Sensitivity + cost-effectiveness analysis
5. Fallback to rule-based narrative if LLM unavailable

Dependencies (add to imports in main.py):
    from db import save_simulation, SimulationRecord
"""

from __future__ import annotations
 
import asyncio
import os
import sys
import uuid
from typing import Optional, AsyncIterator
 
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
 
from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

from db import save_simulation, SimulationRecord

twin_router = APIRouter()
 
 
class TwinStreamRequest(BaseModel):
    patient_state:     dict
    diagnosis:         str = "Unknown diagnosis"
    diagnosis_code:    Optional[str] = None
    treatment_options: list = []
    include_sensitivity_analysis: bool = True
    include_cost_analysis: bool = True

def _extract_ddi_monitoring_context(scenarios: list, recommended_id: str) -> str:
    """
    Build a specific monitoring context string from the recommended scenario's
    safety_check interaction_alerts, so the LLM can generate concrete actions.
    """
    rec_scenario = next((s for s in scenarios if s.get("option_id") == recommended_id), None)
    if not rec_scenario:
        return ""
 
    safety = rec_scenario.get("safety_check") or {}
    interactions = safety.get("interaction_alerts", [])
    allergy_alerts = safety.get("allergy_alerts", [])
 
    if not interactions and not allergy_alerts:
        return ""
 
    lines = []
 
    for ia in interactions:
        proposed = ia.get("proposed_drug", "")
        existing  = ia.get("existing_drug", "")
        warning   = ia.get("warning", "")
 
        # Generate specific monitoring instructions based on known interaction patterns
        if "warfarin" in existing.lower() or "warfarin" in proposed.lower():
            if "azithromycin" in proposed.lower() or "azithromycin" in warning.lower():
                lines.append(
                    f"DDI ({proposed} ↔ {existing}): {warning}. "
                    f"REQUIRED ACTIONS: (1) Check baseline INR before prescribing. "
                    f"(2) Recheck INR at 48-72 hours after starting azithromycin. "
                    f"(3) Hold or reduce warfarin dose if INR >3.5. "
                    f"(4) Target INR 2.0-3.0 for AF."
                )
            elif "levofloxacin" in proposed.lower() or "moxifloxacin" in proposed.lower():
                lines.append(
                    f"DDI ({proposed} ↔ {existing}): {warning}. "
                    f"REQUIRED ACTIONS: (1) Check baseline INR before prescribing. "
                    f"(2) Recheck INR at 3-5 days. "
                    f"(3) Reduce warfarin dose by 20-25% empirically if high-risk patient."
                )
            elif "ceftriaxone" in proposed.lower():
                lines.append(
                    f"DDI ({proposed} ↔ {existing}): {warning}. "
                    f"REQUIRED ACTION: Routine INR monitoring — recheck within 5-7 days."
                )
            else:
                lines.append(
                    f"DDI ({proposed} ↔ {existing}): {warning}. "
                    f"REQUIRED ACTION: Monitor INR closely during co-administration."
                )
        elif "metformin" in existing.lower() or "metformin" in proposed.lower():
            lines.append(
                f"DDI ({proposed} ↔ {existing}): {warning}. "
                f"REQUIRED ACTION: Hold metformin 48h before and after any iodinated contrast."
            )
        else:
            lines.append(f"DDI ({proposed} ↔ {existing}): {warning}.")
 
    for aa in allergy_alerts:
        if aa.get("cross_reactivity"):
            lines.append(
                f"Cross-reactivity alert ({aa.get('drug')} ↔ documented {aa.get('allergen')} allergy, "
                f"{aa.get('severity')} severity): Verify tolerance before prescribing — "
                f"consider allergy skin testing or graded challenge if clinically necessary."
            )
 
    return "\n".join(lines)
 
 
async def _build_narrative_prompt(
    patient_state: dict,
    diagnosis: str,
    diagnosis_code: Optional[str],
    scenarios: list,
    recommended_option: str,
    risk_profile: str,
    sensitivity_top_3: Optional[list],
    cost_effectiveness: Optional[dict],
) -> tuple[ChatPromptTemplate, dict]:
    """
    Bug 4 Fix: Build LLM prompt with DDI-specific monitoring instructions injected.
    The MONITORING section now receives the full interaction context so the LLM
    generates concrete, drug-specific actions rather than generic language.
    """
    demo   = patient_state.get("demographics", {})
    age    = demo.get("age", "?")
    gender = demo.get("gender", "patient")
 
    scenario_lines = []
    for s in scenarios:
        preds     = s.get("predictions", {})
        guideline = s.get("guideline_adherence", {})
        if isinstance(guideline, list):
            guideline = guideline[0] if guideline else {}
 
        adherence_note = ""
        if guideline:
            adherence = guideline.get("adherence", "")
            if adherence == "FIRST_LINE":
                adherence_note = " [FIRST-LINE per guidelines]"
            elif adherence == "SECOND_LINE":
                adherence_note = " [SECOND-LINE]"
            elif adherence == "OFF_GUIDELINE":
                adherence_note = " [OFF-GUIDELINE]"
            elif adherence == "INPATIENT_APPROPRIATE":
                adherence_note = " [INPATIENT guideline]"
 
        safety_flag = (s.get("safety_check") or {}).get("safety_flag", "SAFE")
        safety_note = ""
        if safety_flag == "CONTRAINDICATED":
            safety_note = " [CONTRAINDICATED — excluded from recommendation]"
        elif safety_flag == "INTERACTION_WARNING":
            safety_note = " [DDI warning — see monitoring]"
 
        scenario_lines.append(
            f"Option {s['option_id']} ({s['label']}){adherence_note}{safety_note}: "
            f"7d recovery {preds.get('recovery_probability_7d', 0):.0%}, "
            f"30d mortality {preds.get('mortality_risk_30d', 0):.0%}, "
            f"30d readmission {preds.get('readmission_risk_30d', 0):.0%}"
        )
 
    conditions = patient_state.get("active_conditions", [])
    comorbidity_summary = ", ".join([c.get("display", "") for c in conditions[:3]])
 
    sensitivity_context = ""
    if sensitivity_top_3:
        modifiable = [s for s in sensitivity_top_3 if s.get("modifiable")]
        if modifiable:
            sensitivity_context = "\n\nKey modifiable risk factors:\n" + "\n".join([
                f"- {s['feature_name']} ({s.get('clinical_intervention', 'intervention available')}): "
                f"improvement → {abs(s.get('risk_impact_if_improved', s.get('risk_impact_if_improved_20_percent', {})).get('mortality_30d_change', 0)):.1f}% mortality reduction"
                for s in modifiable[:3]
            ])
 
    cost_context = ""
    if cost_effectiveness:
        most_ce = cost_effectiveness.get("most_cost_effective")
        cost_context = (
            f"\n\nCost-effectiveness: Option {most_ce} is most cost-effective "
            "among treatment options."
        )
 
    # --- Bug 4 fix: inject DDI-specific monitoring context ---
    ddi_monitoring = _extract_ddi_monitoring_context(scenarios, recommended_option)
    monitoring_instruction = (
        f"\n\nDrug interaction monitoring requirements for recommended option:\n{ddi_monitoring}"
        if ddi_monitoring else
        "\n\nNo significant drug interactions for recommended option."
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical decision support system generating structured clinical decision support notes. "
         "Use the following format exactly:\n\n"
         "IMPRESSION: One sentence summarizing patient risk profile and primary diagnosis.\n\n"
         "SAFETY: One sentence. If any option is CONTRAINDICATED, name it explicitly, state why "
         "(allergy/cross-reactivity/DDI), and confirm it is excluded from recommendation.\n\n"
         "RECOMMENDATION: One to two sentences. State the recommended option, cite its key outcome "
         "metrics (7d recovery, 30d mortality, readmission), and note guideline adherence status.\n\n"
         "MONITORING: Two to three sentences. Use the 'Drug interaction monitoring requirements' provided "
         "below — cite the specific drugs, the specific INR targets, specific timing of rechecks, and "
         "any dose adjustment thresholds. Do NOT write generic phrases like 'monitor closely' — "
         "write the exact clinical actions (e.g. 'Check baseline INR before prescribing azithromycin, "
         "recheck at 48-72h, hold warfarin if INR >3.5, target INR 2.0-3.0 for atrial fibrillation').\n\n"
         "MODIFIABLE RISK FACTORS: One sentence. Name 1-2 specific modifiable lab factors with their "
         "quantified mortality impact from the sensitivity data. Do NOT mention age.\n\n"
         "CRITICAL RULES:\n"
         "- CONTRAINDICATED options must never appear as recommendations or alternatives.\n"
         "- All percentages must come from the simulation data provided — do not invent numbers.\n"
         "- MONITORING must reference the specific drug names and specific clinical thresholds given.\n"
         "- End with exactly: 'This is AI-generated decision support requiring physician validation.'"),
        ("human",
         f"Patient: {age}y {gender}. "
         f"Diagnosis: {diagnosis} (ICD-10: {diagnosis_code or 'not specified'}). "
         f"Comorbidities: {comorbidity_summary or 'none documented'}. "
         f"Risk profile: {risk_profile}.\n\n"
         f"Treatment scenarios:\n" + "\n".join(scenario_lines) + "\n\n"
         f"Recommended: Option {recommended_option}"
         f"{sensitivity_context}"
         f"{cost_context}"
         f"{monitoring_instruction}\n\n"
         "Generate the structured clinical decision support note."),
    ])
 
    return prompt, {}


async def _twin_stream(req: TwinStreamRequest) -> AsyncIterator[str]:
    node  = "digital_twin"
    timer = Timer()
    request_id = str(uuid.uuid4())
 
    # Import from main module
    from main import (
        _models, _models_loaded, _llm, _llm_ready, _tools_ready,
        predict_baseline_risks_with_uncertainty,
        _determine_model_confidence, sanitize,
    )
    from simulator import (
        simulate_treatment, determine_patient_risk_profile, select_recommended_option,
    )
    from feature_engineering import engineer_features, FEATURE_NAMES, get_feature_attribution
    from clinical_tools import (
        check_drug_guideline_adherence,
        perform_sensitivity_analysis, analyze_cost_effectiveness,
        build_enhanced_fhir_care_plan,
    )
    from model import TreatmentOption
 
    if not _models_loaded:
        yield evt_error(node, "XGBoost models not loaded — run train_models.py first", fatal=True)
        return
 
    # ── Step 1: Feature Engineering ───────────────────────────────────────────
    yield evt_status(node, "Engineering patient features...", step=1, total=7)
    try:
        feature_vector, feature_dict = await asyncio.to_thread(
            engineer_features, req.patient_state
        )
    except Exception as exc:
        yield evt_error(node, f"Feature engineering failed: {exc}", fatal=True)
        return
 
    yield evt_progress(node, f"{len(feature_vector)} features extracted", pct=12)
 
    # ── Step 2: XGBoost Risk Prediction ───────────────────────────────────────
    yield evt_status(node, "Running XGBoost risk models with uncertainty quantification...",
                     step=2, total=7)
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
    risk_profile = determine_patient_risk_profile(baseline_risks)
    model_confidence = _determine_model_confidence(baseline_risks_ci)
 
    yield evt_progress(node,
                       f"Risk profile: {risk_profile} | "
                       f"Mortality 30d: {baseline_risks['mortality_30d']:.0%} "
                       f"(CI: {baseline_risks_ci['mortality_30d']['lower_bound_95ci']:.0%}-"
                       f"{baseline_risks_ci['mortality_30d']['upper_bound_95ci']:.0%})",
                       pct=35)
 
    # ── Step 3: Treatment Options ─────────────────────────────────────────────
    treatment_options = [TreatmentOption(**o) if isinstance(o, dict) else o
                         for o in req.treatment_options]
    if not any(o.option_id == "C" for o in treatment_options):
        treatment_options.append(TreatmentOption(
            option_id="C", label="No treatment (baseline)", drugs=[], interventions=[]
        ))
 
    diagnosis_code = req.diagnosis_code or req.diagnosis.split("(")[-1].strip(")")
 
    # ── Step 4: Treatment Simulation ──────────────────────────────────────────
    yield evt_status(node, "Simulating treatment scenarios...", step=3, total=7)
    
    scenarios = []
    for opt in treatment_options:
        preds = await asyncio.to_thread(
            simulate_treatment, baseline_risks, opt.drugs, opt.interventions,
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
 
    yield evt_progress(node, f"{len(scenarios)} scenarios simulated", pct=50)
 
    # ── Step 5: Recommendation ────────────────────────────────────────────────
    scoreable = [s for s in scenarios if s["option_id"] != "C"]
    recommended_id, rec_confidence = (
        select_recommended_option(scoreable) if scoreable else ("A", 0.70)
    )
 
    yield evt_progress(node, f"Recommended: Option {recommended_id} ({rec_confidence:.0%} confidence)",
                       pct=60)
 
    # ── Step 6: Sensitivity Analysis ──────────────────────────────────────────
    sensitivity_results = None
    if req.include_sensitivity_analysis:
        yield evt_status(node, "Performing sensitivity analysis...", step=4, total=7)
        try:
            sensitivity_results = await asyncio.to_thread(
                perform_sensitivity_analysis,
                feature_vector, feature_dict, _models, FEATURE_NAMES
            )
            yield evt_progress(node, f"Identified {len(sensitivity_results)} modifiable factors", pct=65)
        except Exception as e:
            yield evt_error(node, f"Sensitivity analysis failed: {e}", fatal=False)
 
    # ── Step 7: Cost-Effectiveness ────────────────────────────────────────────
    cost_effectiveness = None
    if req.include_cost_analysis:
        yield evt_status(node, "Analyzing cost-effectiveness...", step=5, total=7)
        try:
            patient_age = req.patient_state.get("demographics", {}).get("age", 65)
            cost_effectiveness = await asyncio.to_thread(
                analyze_cost_effectiveness, scenarios, patient_age
            )
            yield evt_progress(node, "Cost-effectiveness analysis complete", pct=70)
        except Exception as e:
            yield evt_error(node, f"Cost-effectiveness failed: {e}", fatal=False)
 
    # ── Step 8: LLM Narrative Streaming ───────────────────────────────────────
    yield evt_status(node, "Generating clinical narrative (streaming LLM tokens)...",
                     step=6, total=7)
 
    narrative = ""
    token_count = 0
    
    if _llm_ready and _llm:
        try:
            prompt, inputs = await _build_narrative_prompt(
                req.patient_state, req.diagnosis, diagnosis_code,
                scenarios, recommended_id, risk_profile,
                sensitivity_results[:3] if sensitivity_results else None,
                cost_effectiveness,
            )
            
            chain = prompt | _llm | StrOutputParser()
            
            # Stream tokens using astream_events (v2 API)
            async for event in chain.astream_events(inputs, version="v2"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        narrative += chunk.content
                        yield evt_token(node, chunk.content)
                
                elif kind == "on_chain_end" and event["name"] == "StrOutputParser":
                    raw = event["data"].get("output")
                    if isinstance(raw, str):
                        narrative = raw
            
            if "physician validation" not in narrative.lower():
                narrative += " This is AI-generated decision support requiring physician validation."
            
            yield evt_progress(node, f"Narrative generated ({token_count} tokens)", pct=85)
        
        except Exception as exc:
            yield evt_error(node, f"LLM streaming failed: {exc} — using fallback", fatal=False)
            narrative = (
                f"For this patient with {req.diagnosis} ({risk_profile} risk), "
                f"recommended treatment offers clinical benefit. "
                "Evidence-based decision support — clinical judgment required."
            )
    else:
        yield evt_progress(node, "LLM unavailable — using rule-based narrative", pct=85)
        narrative = (
            f"For this patient with {req.diagnosis} ({risk_profile} risk), "
            f"recommended treatment offers clinical benefit. "
            "Evidence-based decision support — clinical judgment required."
        )
 
    # ── Step 9: Feature Attribution + FHIR ────────────────────────────────────
    attribution = await asyncio.to_thread(get_feature_attribution, feature_dict, baseline_risks)
 
    yield evt_status(node, "Building FHIR CarePlan...", step=7, total=7)
    
    rec_opt = next((o for o in treatment_options if o.option_id == recommended_id),
                   treatment_options[0])
    rec_scen = next((s for s in scenarios if s["option_id"] == recommended_id), scenarios[0])
    
    fhir_care_plan = await asyncio.to_thread(
        build_enhanced_fhir_care_plan,
        req.patient_state.get("patient_id", "unknown"),
        rec_opt, narrative,
        rec_scen["predictions"].get("recovery_probability_7d", 0.7),
        model_confidence, diagnosis_code, attribution,
        model_version="2.0.0",
    )
 
    # ── Assemble Result ───────────────────────────────────────────────────────
    result = sanitize({
        "simulation_summary": {
            "patient_risk_profile": risk_profile,
            "baseline_risks": {k: round(float(v), 3) for k, v in baseline_risks.items()},
            "baseline_risks_with_ci": baseline_risks_ci,
            "recommended_option": recommended_id,
            "recommendation_confidence": float(rec_confidence),
            "model_confidence": model_confidence,
        },
        "scenarios": scenarios,
        "what_if_narrative": narrative,
        "fhir_care_plan": fhir_care_plan,
        "feature_attribution": attribution,
        "sensitivity_analysis": sensitivity_results,
        "cost_effectiveness_summary": cost_effectiveness,
        "models_loaded": True,
        "model_confidence": model_confidence,
        "mock": False,
        "provenance": {
            "simulation_id": request_id,
            "llm_tokens_generated": token_count,
        },
    })
 
    # ── Database Persistence ──────────────────────────────────────────────────
    patient_id = req.patient_state.get("patient_id", "unknown")
    
    await save_simulation(SimulationRecord(
        request_id=request_id,
        patient_id=patient_id,
        diagnosis=req.diagnosis,
        diagnosis_code=diagnosis_code,
        patient_risk_profile=risk_profile,
        baseline_mortality_30d=baseline_risks["mortality_30d"],
        baseline_readmission_30d=baseline_risks["readmission_30d"],
        baseline_complication=baseline_risks["complication"],
        recommended_option=recommended_id,
        recommendation_confidence=rec_confidence,
        model_confidence=model_confidence,
        treatment_options_count=len(treatment_options),
        scenarios=scenarios,
        simulation_summary=result["simulation_summary"],
        what_if_narrative=narrative,
        fhir_care_plan=fhir_care_plan,
        feature_attribution=attribution,
        sensitivity_analysis=sensitivity_results,
        cost_effectiveness=cost_effectiveness,
        models_loaded=True,
        cache_hit=False,
        elapsed_ms=timer.elapsed_ms(),
        source="stream",
    ))
 
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@twin_router.post("/stream")
async def twin_stream(request: TwinStreamRequest):
    """SSE streaming endpoint with LLM narrative token streaming."""
    async def gen():
        async for chunk in _twin_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)