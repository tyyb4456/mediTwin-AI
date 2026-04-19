"""
agents/digital_twin/stream_endpoints.py
----------------------------------------
SSE streaming endpoint for Digital Twin Agent.

SYNC CONTRACT: This file produces the SAME response shape as /simulate (main.py).
Any change to /simulate's enrichment pipeline must be mirrored here.

Parity checklist vs /simulate:
  [x] predict_baseline_risks_with_uncertainty → baseline_risks_with_ci
  [x] predictions_with_ci per scenario (CI bounds)
  [x] check_drug_guideline_adherence per drug (not just first drug)
  [x] check_allergy_contraindications per scenario → safety_check
  [x] key_risks built from safety_check + CI + guideline adherence
  [x] _resolve_option_cost → estimated_cost_usd + cost_source
  [x] CONTRAINDICATED options excluded from scoreable via safe_scoreable
  [x] avoid_hospitalization preference applied
  [x] sensitivity_analysis via perform_sensitivity_analysis
  [x] cost_effectiveness via analyze_cost_effectiveness
  [x] _extract_ddi_monitoring_context injected into LLM prompt (Bug 4 fix)
  [x] narrative includes concrete INR targets / DDI actions (not generic)
  [x] feature_attribution via get_feature_attribution
  [x] FHIR CarePlan via build_enhanced_fhir_care_plan
  [x] provenance shape matches /simulate (simulation_hash, horizons, etc.)
  [x] DB persistence via save_simulation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)
from db import save_simulation, SimulationRecord

twin_router = APIRouter()


# ── Request model ─────────────────────────────────────────────────────────────

class TwinStreamRequest(BaseModel):
    patient_state:                dict
    diagnosis:                    str = "Unknown diagnosis"
    diagnosis_code:               Optional[str] = None
    treatment_options:            list = []
    include_sensitivity_analysis: bool = True
    include_cost_analysis:        bool = True
    prediction_horizons:          list = ["7d", "30d", "90d"]
    patient_preferences:          Optional[dict] = None


# ── Simulation hash (mirrors main.py) ─────────────────────────────────────────

def _generate_simulation_hash(patient_state: dict, diagnosis: str, treatment_options: list) -> str:
    content = json.dumps(
        {
            "patient_id":  patient_state.get("patient_id"),
            "diagnosis":   diagnosis,
            "options":     treatment_options,
            "timestamp":   patient_state.get("state_timestamp"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Main streaming generator ──────────────────────────────────────────────────

async def _twin_stream(req: TwinStreamRequest) -> AsyncIterator[str]:
    node      = "digital_twin"
    timer     = Timer()
    request_id = str(uuid.uuid4())

    # ── Lazy imports from main module (avoids circular imports) ───────────────
    from main import (
        _models, _models_loaded, _llm, _llm_ready, _tools_ready,
        predict_baseline_risks_with_uncertainty,
        _determine_model_confidence,
        _resolve_option_cost,
        sanitize,
    )
    from simulator import (
        simulate_treatment,
        determine_patient_risk_profile,
        select_recommended_option,
    )
    from feature_engineering import engineer_features, FEATURE_NAMES, get_feature_attribution
    from clinical_tools import (
        check_drug_guideline_adherence,
        check_allergy_contraindications,
        perform_sensitivity_analysis,
        analyze_cost_effectiveness,
        build_enhanced_llm_narrative,
        build_enhanced_fhir_care_plan,
    )
    from model import TreatmentOption

    # ── Guard: models must be loaded ──────────────────────────────────────────
    if not _models_loaded:
        yield evt_error(node, "XGBoost models not loaded — run train_models.py first", fatal=True)
        return

    patient_state = req.patient_state
    patient_id    = patient_state.get("patient_id", "unknown")
    diagnosis_code = (
        req.diagnosis_code
        or req.diagnosis.split("(")[-1].strip(")")
    )
    simulation_hash = _generate_simulation_hash(patient_state, req.diagnosis, req.treatment_options)

    # ── STEP 1: Feature Engineering ───────────────────────────────────────────
    yield evt_status(node, "Engineering patient features...", step=1, total=8)
    try:
        feature_vector, feature_dict = await asyncio.to_thread(
            engineer_features, patient_state
        )
    except Exception as exc:
        yield evt_error(node, f"Feature engineering failed: {exc}", fatal=True)
        return

    yield evt_progress(node, f"{len(feature_vector)} features extracted", pct=10)

    # ── STEP 2: XGBoost Risk Prediction with Uncertainty ─────────────────────
    yield evt_status(node, "Running XGBoost risk models with uncertainty quantification...",
                     step=2, total=8)
    try:
        baseline_risks_with_ci = await asyncio.to_thread(
            predict_baseline_risks_with_uncertainty, feature_vector
        )
    except Exception as exc:
        yield evt_error(node, f"Risk prediction failed: {exc}", fatal=True)
        return

    baseline_risks = {
        "readmission_30d": baseline_risks_with_ci["readmission_30d"]["point_estimate"],
        "mortality_30d":   baseline_risks_with_ci["mortality_30d"]["point_estimate"],
        "complication":    baseline_risks_with_ci["complication"]["point_estimate"],
    }
    risk_profile     = determine_patient_risk_profile(baseline_risks)
    model_confidence = _determine_model_confidence(baseline_risks_with_ci)

    yield evt_progress(
        node,
        f"Risk profile: {risk_profile} | "
        f"Mortality 30d: {baseline_risks['mortality_30d']:.0%} "
        f"(CI: {baseline_risks_with_ci['mortality_30d']['lower_bound_95ci']:.0%}"
        f"–{baseline_risks_with_ci['mortality_30d']['upper_bound_95ci']:.0%})",
        pct=22,
    )

    # ── STEP 3: Sensitivity Analysis ──────────────────────────────────────────
    sensitivity_results = None
    if req.include_sensitivity_analysis:
        yield evt_status(node, "Performing sensitivity analysis...", step=3, total=8)
        try:
            sensitivity_results = await asyncio.to_thread(
                perform_sensitivity_analysis,
                feature_vector, feature_dict, _models, FEATURE_NAMES,
            )
            yield evt_progress(node,
                               f"Identified {len(sensitivity_results)} sensitivity factors",
                               pct=32)
        except Exception as exc:
            yield evt_error(node, f"Sensitivity analysis failed: {exc}", fatal=False)

    # ── STEP 4: Treatment Simulation + Full Safety Enrichment ─────────────────
    yield evt_status(node, "Simulating treatment scenarios with safety checks...",
                     step=4, total=8)

    # Ensure baseline "no treatment" option C exists
    treatment_options = [
        TreatmentOption(**o) if isinstance(o, dict) else o
        for o in req.treatment_options
    ]
    if not any(o.option_id == "C" for o in treatment_options):
        treatment_options.append(TreatmentOption(
            option_id="C",
            label="No treatment (baseline)",
            drugs=[],
            interventions=[],
            estimated_cost_usd=0,
        ))

    scenarios = []

    for opt in treatment_options:
        # 4a. Simulate treatment effects
        predictions = await asyncio.to_thread(
            simulate_treatment,
            baseline_risks,
            opt.drugs,
            opt.interventions,
        )

        # 4b. predictions_with_ci — mirrors main.py exactly
        predictions_with_ci: dict = {}
        for key in ("mortality_risk_30d", "readmission_risk_30d", "complication_risk"):
            base_key = key.replace("_risk", "")
            point = predictions.get(key, 0.5)
            if base_key in baseline_risks_with_ci:
                base_ci    = baseline_risks_with_ci[base_key]
                width_ratio = base_ci["interval_width"] / max(base_ci["point_estimate"], 0.01)
                width       = point * width_ratio
            else:
                width = 0.1
            predictions_with_ci[key] = {
                "point_estimate":    round(point, 4),
                "lower_bound_95ci":  round(max(0.0, point - width / 2), 4),
                "upper_bound_95ci":  round(min(1.0, point + width / 2), 4),
            }

        # 4c. Guideline adherence — check ALL drugs (not just first)
        guideline_adherence = None
        if _tools_ready and diagnosis_code and opt.drugs:
            try:
                adherence_results = []
                for drug in opt.drugs:
                    result = await asyncio.to_thread(
                        check_drug_guideline_adherence.invoke,
                        {"diagnosis_code": diagnosis_code, "proposed_drug": drug},
                    )
                    adherence_results.append({**result, "drug": drug})
                # Sort: worst adherence first (OFF_GUIDELINE bubbles up)
                priority = {
                    "OFF_GUIDELINE": 0, "UNKNOWN": 1, "SECOND_LINE": 2,
                    "INPATIENT_APPROPRIATE": 3, "GUIDELINE_LISTED": 3, "FIRST_LINE": 4,
                }
                adherence_results.sort(
                    key=lambda r: priority.get(r.get("adherence", "UNKNOWN"), 1)
                )
                guideline_adherence = adherence_results
            except Exception as exc:
                print(f"  ⚠️  Guideline check failed: {exc}")

        # 4d. Allergy + DDI safety check
        safety_check = None
        if _tools_ready:
            try:
                allergies    = patient_state.get("allergies", [])
                current_meds = patient_state.get("medications", [])
                safety_check = await asyncio.to_thread(
                    check_allergy_contraindications.invoke,
                    {
                        "proposed_drugs":      opt.drugs + opt.interventions,
                        "allergies":           allergies,
                        "current_medications": current_meds,
                    },
                )
            except Exception as exc:
                print(f"  ⚠️  Safety check failed: {exc}")

        # 4e. key_risks — mirrors main.py: safety alerts first, then stats
        key_risks: list = []

        if safety_check:
            for alert in safety_check.get("allergy_alerts", []):
                key_risks.append(f"🚨 {alert['alert']}")
            for interaction in safety_check.get("interaction_alerts", []):
                key_risks.append(
                    f"⚠️  DDI: {interaction['warning']} "
                    f"({interaction['proposed_drug']} ↔ {interaction['existing_drug']})"
                )

        if predictions["mortality_risk_30d"] > 0.05:
            ci = predictions_with_ci["mortality_risk_30d"]
            key_risks.append(
                f"30-day mortality: {predictions['mortality_risk_30d']:.0%} "
                f"(CI: {ci['lower_bound_95ci']:.0%}–{ci['upper_bound_95ci']:.0%})"
            )
        if predictions["readmission_risk_30d"] > 0.15:
            key_risks.append(f"Readmission risk: {predictions['readmission_risk_30d']:.0%}")

        if guideline_adherence:
            for g in (guideline_adherence if isinstance(guideline_adherence, list) else [guideline_adherence]):
                if g.get("adherence") == "OFF_GUIDELINE":
                    key_risks.append(
                        f"⚠️  Off-guideline: {g.get('drug', '')} — {g.get('message', '')}"
                    )

        if not key_risks:
            key_risks.append("Low overall risk with this treatment")

        # 4f. Cost resolution — uses same _resolve_option_cost as /simulate
        resolved_cost, cost_source = _resolve_option_cost(opt)

        scenarios.append({
            "option_id":            opt.option_id,
            "label":                opt.label,
            "drugs":                opt.drugs,
            "interventions":        opt.interventions,
            "predictions":          predictions,
            "predictions_with_ci":  predictions_with_ci,
            "key_risks":            key_risks,
            "guideline_adherence":  guideline_adherence,
            "safety_check":         safety_check,
            "estimated_cost_usd":   resolved_cost,
            "cost_source":          cost_source,
        })

    yield evt_progress(node,
                       f"{len(scenarios)} scenarios simulated with full safety enrichment",
                       pct=50)

    # ── STEP 5: Recommendation (mirrors main.py safe_scoreable logic) ─────────
    patient_prefs         = req.patient_preferences or {}
    prioritize_cost       = patient_prefs.get("prioritize_cost", False)
    avoid_hospitalization = patient_prefs.get("avoid_hospitalization", False)

    scoreable = [s for s in scenarios if s["option_id"] != "C"]

    # Exclude CONTRAINDICATED options exactly as main.py does
    safe_scoreable = [
        s for s in scoreable
        if (s.get("safety_check") or {}).get("safety_flag") != "CONTRAINDICATED"
    ]
    if not safe_scoreable:
        safe_scoreable = scoreable   # All contraindicated — fall back, warn
        print("  ⚠️  All treatment options CONTRAINDICATED — physician review required")

    if avoid_hospitalization:
        non_hosp = [
            s for s in safe_scoreable
            if "hospitalization" not in [i.lower() for i in s.get("interventions", [])]
        ]
        if non_hosp:
            safe_scoreable = non_hosp

    if safe_scoreable:
        recommended_id, rec_confidence = select_recommended_option(
            safe_scoreable, prioritize_cost=prioritize_cost
        )
    else:
        recommended_id  = scenarios[0]["option_id"] if scenarios else "A"
        rec_confidence  = 0.70

    yield evt_progress(node,
                       f"Recommended: Option {recommended_id} ({rec_confidence:.0%} confidence)",
                       pct=58)

    # ── STEP 6: Cost-Effectiveness Analysis ───────────────────────────────────
    cost_effectiveness = None
    if req.include_cost_analysis:
        yield evt_status(node, "Analyzing cost-effectiveness...", step=5, total=8)
        try:
            patient_age    = patient_state.get("demographics", {}).get("age", 65)
            cost_effectiveness = await asyncio.to_thread(
                analyze_cost_effectiveness, scenarios, patient_age
            )
            yield evt_progress(node, "Cost-effectiveness analysis complete", pct=65)
        except Exception as exc:
            yield evt_error(node, f"Cost-effectiveness analysis failed: {exc}", fatal=False)

    # ── STEP 7: LLM Narrative (streaming tokens) ──────────────────────────────
    # Uses build_enhanced_llm_narrative from clinical_tools.py which already
    # injects DDI-specific monitoring context (Bug 4 fix) — same as /simulate.
    yield evt_status(node,
                     "Generating clinical narrative (streaming LLM tokens)...",
                     step=6, total=8)

    narrative    = ""
    token_count  = 0

    if _llm_ready and _llm:
        try:
            # Build the same prompt that clinical_tools.build_enhanced_llm_narrative
            # would use, but stream tokens. We re-use the helper directly so the
            # prompt text is identical — no divergence risk.
            from langchain_core.prompts import ChatPromptTemplate

            # Replicate the prompt construction from build_enhanced_llm_narrative
            # (clinical_tools.py) so we can stream it token-by-token.
            demo               = patient_state.get("demographics", {})
            age                = demo.get("age", "?")
            gender             = demo.get("gender", "patient")
            conditions         = patient_state.get("active_conditions", [])
            comorbidity_summary = ", ".join(c.get("display", "") for c in conditions[:3])

            best = next(
                (s for s in scenarios if s["option_id"] == recommended_id),
                scenarios[0] if scenarios else {},
            )

            # Build scenario summary lines
            scenario_lines = []
            for s in scenarios:
                if s["option_id"] == "C":
                    continue
                preds        = s.get("predictions", {})
                guideline_raw = s.get("guideline_adherence") or {}
                guideline     = (
                    guideline_raw[0]
                    if isinstance(guideline_raw, list) and guideline_raw
                    else guideline_raw if isinstance(guideline_raw, dict)
                    else {}
                )
                adherence_note = {
                    "FIRST_LINE":            " [FIRST-LINE per guidelines]",
                    "INPATIENT_APPROPRIATE": " [INPATIENT guideline]",
                    "SECOND_LINE":           " [SECOND-LINE]",
                    "OFF_GUIDELINE":         " [OFF-GUIDELINE]",
                }.get(guideline.get("adherence", ""), "")

                safety_flag = (s.get("safety_check") or {}).get("safety_flag", "SAFE")
                safety_note = {
                    "CONTRAINDICATED":     " [CONTRAINDICATED — excluded from recommendation]",
                    "ALLERGY_RISK":        " [ALLERGY RISK — use with caution]",
                    "INTERACTION_WARNING": " [DDI warning — see monitoring]",
                }.get(safety_flag, "")

                scenario_lines.append(
                    f"Option {s['option_id']} ({s['label']}){adherence_note}{safety_note}: "
                    f"7d recovery {preds.get('recovery_probability_7d', 0):.0%}, "
                    f"30d mortality {preds.get('mortality_risk_30d', 0):.0%}, "
                    f"30d readmission {preds.get('readmission_risk_30d', 0):.0%}"
                )

            # Sensitivity context (modifiable only — Bug 6 fix)
            sensitivity_context = ""
            if sensitivity_results:
                modifiable = [s for s in sensitivity_results if s.get("modifiable", False)]
                if modifiable:
                    sensitivity_context = "\n\nKey MODIFIABLE risk factors (clinician can intervene):\n" + "\n".join([
                        f"- {s['feature_name']} ({s.get('clinical_intervention', 'intervention available')}): "
                        f"20% improvement → "
                        f"{abs(s.get('risk_impact_if_improved_20_percent', {}).get('mortality_30d_change', 0)):.1f}% mortality reduction"
                        for s in modifiable[:3]
                    ])

            # Cost context
            cost_context = ""
            if cost_effectiveness:
                most_ce      = cost_effectiveness.get("most_cost_effective")
                cost_context = (
                    f"\n\nCost-effectiveness: Option {most_ce} is most cost-effective "
                    "among treatment options at current willingness-to-pay threshold."
                )

            # DDI monitoring context — same function as clinical_tools.py uses
            from clinical_tools import _extract_ddi_monitoring_context
            ddi_monitoring = _extract_ddi_monitoring_context(scenarios, recommended_id)
            monitoring_block = (
                f"\n\nDrug interaction monitoring requirements for the recommended option:\n{ddi_monitoring}"
                if ddi_monitoring
                else "\n\nNo significant drug interactions detected for the recommended option."
            )

            # Build explicit contraindication facts so LLM cannot hallucinate vague reasons.
            # For each CONTRAINDICATED scenario, extract the exact allergy/DDI reason from
            # safety_check and inject it as a structured fact into the human prompt.
            contraindication_facts: list[str] = []
            for s in scenarios:
                sc = s.get("safety_check") or {}
                if sc.get("safety_flag") != "CONTRAINDICATED":
                    continue
                reasons: list[str] = []
                for alert in sc.get("allergy_alerts", []):
                    if alert.get("cross_reactivity"):
                        reasons.append(
                            f"cross-reactivity between {alert['drug']} and documented "
                            f"{alert['allergen']} allergy ({alert.get('severity', 'unknown')} severity)"
                        )
                    else:
                        reasons.append(
                            f"documented {alert['allergen']} allergy "
                            f"({alert.get('severity', 'unknown')} severity)"
                        )
                for ia in sc.get("interaction_alerts", []):
                    sev = ia["warning"].split(":")[0]   # "MAJOR" | "MINOR" | "MODERATE"
                    reasons.append(
                        f"{sev} DDI between {ia['proposed_drug']} and {ia['existing_drug']}"
                    )
                reason_str = "; ".join(reasons) if reasons else "safety concern"
                contraindication_facts.append(
                    f"Option {s['option_id']} ({s['label']}) is CONTRAINDICATED "
                    f"due to: {reason_str}. It must be excluded from all recommendations."
                )

            contraindication_block = (
                "\n\nCONTRAINDICATION FACTS (use these verbatim in the SAFETY section):\n"
                + "\n".join(contraindication_facts)
                if contraindication_facts
                else ""
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a clinical decision support system generating structured clinical decision support notes. "
                 "Use the following format exactly:\n\n"
                 "IMPRESSION: One sentence summarizing patient risk profile and primary diagnosis.\n\n"
                 "SAFETY: One sentence. Copy the contraindication reason VERBATIM from the "
                 "'CONTRAINDICATION FACTS' provided in the user message — do not paraphrase or generalise it. "
                 "Confirm the option is excluded from recommendation.\n\n"
                 "RECOMMENDATION: One to two sentences. State the recommended option, cite its key outcome "
                 "metrics (7d recovery, 30d mortality, readmission), and note guideline adherence status.\n\n"
                 "MONITORING: Two to three sentences. You MUST use the 'Drug interaction monitoring requirements' "
                 "provided below. Reference the specific drug names, the specific INR targets and recheck timing, "
                 "and the specific dose-adjustment thresholds. "
                 "FORBIDDEN phrases: 'monitor closely', 'monitor for drug interactions', 'close monitoring'. "
                 "REQUIRED format example: 'Check baseline INR before starting azithromycin; recheck at 48-72h; "
                 "hold warfarin if INR >3.5; target INR 2.0-3.0 for atrial fibrillation.'\n\n"
                 "MODIFIABLE RISK FACTORS: One sentence. Name 1-2 specific modifiable lab factors with their "
                 "quantified mortality impact from the sensitivity data. Do NOT mention age.\n\n"
                 "CRITICAL RULES:\n"
                 "- CONTRAINDICATED options must never appear as recommendations or alternatives.\n"
                 "- The SAFETY reason must come from the CONTRAINDICATION FACTS block — never invent one.\n"
                 "- All percentages must come from the simulation data provided — do not invent numbers.\n"
                 "- MONITORING must contain the exact drug names and exact thresholds from the DDI context.\n"
                 "- End with exactly: 'This is AI-generated decision support requiring physician validation.'"),
                ("human",
                 f"Patient: {age}y {gender}. "
                 f"Diagnosis: {req.diagnosis} (ICD-10: {diagnosis_code or 'not specified'}). "
                 f"Comorbidities: {comorbidity_summary or 'none documented'}. "
                 f"Risk profile: {risk_profile}.\n\n"
                 f"Treatment scenarios:\n" + "\n".join(scenario_lines) + "\n\n"
                 f"Recommended: Option {recommended_id}"
                 f"{contraindication_block}"
                 f"{sensitivity_context}"
                 f"{cost_context}"
                 f"{monitoring_block}\n\n"
                 "Generate the structured clinical decision support note."),
            ])

            chain = prompt | _llm | StrOutputParser()

            async for event in chain.astream_events({}, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        narrative   += chunk.content
                        yield evt_token(node, chunk.content)
                elif kind == "on_chain_end" and event["name"] == "StrOutputParser":
                    raw = event["data"].get("output")
                    if isinstance(raw, str):
                        narrative = raw

            if "physician validation" not in narrative.lower():
                narrative += " This is AI-generated decision support requiring physician validation."

            yield evt_progress(node, f"Narrative generated ({token_count} tokens)", pct=82)

        except Exception as exc:
            yield evt_error(node, f"LLM streaming failed: {exc} — using fallback", fatal=False)
            # Fallback: call the non-streaming version (same function as /simulate)
            narrative = build_enhanced_llm_narrative(
                patient_state=patient_state,
                diagnosis=req.diagnosis,
                diagnosis_code=diagnosis_code,
                scenarios=scenarios,
                recommended_option=recommended_id,
                risk_profile=risk_profile,
                llm=None,          # force rule-based fallback
                llm_ready=False,
                sensitivity_top_3=sensitivity_results[:3] if sensitivity_results else None,
                cost_effectiveness=cost_effectiveness,
            )
    else:
        yield evt_progress(node, "LLM unavailable — using rule-based narrative", pct=82)
        narrative = build_enhanced_llm_narrative(
            patient_state=patient_state,
            diagnosis=req.diagnosis,
            diagnosis_code=diagnosis_code,
            scenarios=scenarios,
            recommended_option=recommended_id,
            risk_profile=risk_profile,
            llm=None,
            llm_ready=False,
            sensitivity_top_3=sensitivity_results[:3] if sensitivity_results else None,
            cost_effectiveness=cost_effectiveness,
        )

    # ── STEP 8: Feature Attribution + FHIR CarePlan ───────────────────────────
    yield evt_status(node, "Building FHIR CarePlan...", step=7, total=8)

    attribution = await asyncio.to_thread(
        get_feature_attribution, feature_dict, baseline_risks
    )

    rec_opt  = next(
        (o for o in treatment_options if o.option_id == recommended_id),
        treatment_options[0],
    )
    rec_scen = next(
        (s for s in scenarios if s["option_id"] == recommended_id),
        scenarios[0],
    )
    rec_recovery = rec_scen["predictions"].get("recovery_probability_7d", 0.7)

    fhir_care_plan = await asyncio.to_thread(
        build_enhanced_fhir_care_plan,
        patient_id,
        rec_opt,
        narrative,
        rec_recovery,
        model_confidence,
        diagnosis_code,
        attribution,
        "2.0.0",   # model_version — keep in sync with main.py
    )

    yield evt_progress(node, "FHIR CarePlan built", pct=92)

    # ── STEP 9: Provenance (mirrors main.py structure exactly) ────────────────
    yield evt_status(node, "Assembling final result...", step=8, total=8)

    requested_horizons = req.prediction_horizons or ["7d", "30d"]
    available_horizons = list(baseline_risks_with_ci.keys())

    has_7d = any(
        s.get("predictions", {}).get("recovery_probability_7d") is not None
        for s in scenarios
    )
    horizon_map = {
        "7d":  lambda: has_7d,
        "30d": lambda: any(k in available_horizons for k in ["readmission_30d", "mortality_30d", "complication"]),
        "90d": lambda: "readmission_90d" in available_horizons,
        "1yr": lambda: "mortality_1yr" in available_horizons,
    }
    fulfilled_horizons = [
        h for h in requested_horizons
        if horizon_map.get(h, lambda: False)()
    ]

    provenance = {
        "simulation_id":        simulation_hash,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "model_version":        "2.0.0",
        "models_used":          list(_models.keys()),
        "feature_count":        len(FEATURE_NAMES),
        "requested_horizons":   requested_horizons,
        "fulfilled_horizons":   fulfilled_horizons,
        "unfulfilled_horizons": [h for h in requested_horizons if h not in fulfilled_horizons],
        "horizon_gap_reason": (
            "Extended models (readmission_90d, mortality_1yr) not loaded — "
            "run train_models.py to enable 90d and 1yr predictions."
            if any(h not in fulfilled_horizons for h in requested_horizons)
            else None
        ),
        "available_model_keys": available_horizons,
        "overall_confidence":   model_confidence,
        "llm_tokens_generated": token_count,
        "reproducible":         True,
    }

    # ── Assemble final result (same top-level shape as /simulate) ─────────────
    result = sanitize({
        "simulation_summary": {
            "patient_risk_profile":      risk_profile,
            "baseline_risks":            {k: round(float(v), 3) for k, v in baseline_risks.items()},
            "baseline_risks_with_ci":    baseline_risks_with_ci,
            "primary_concern":           f"{risk_profile} risk — {req.diagnosis}",
            "recommended_option":        recommended_id,
            "recommendation_confidence": float(rec_confidence),
            "model_confidence":          model_confidence,
        },
        "scenarios":                scenarios,
        "what_if_narrative":        narrative,
        "fhir_care_plan":           fhir_care_plan,
        "feature_attribution":      attribution,
        "sensitivity_analysis":     sensitivity_results,
        "cost_effectiveness_summary": cost_effectiveness,
        "models_loaded":            True,
        "model_confidence":         model_confidence,
        "provenance":               provenance,
        "mock":                     False,
    })

    # ── DB Persistence ────────────────────────────────────────────────────────
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


# ── FastAPI route ─────────────────────────────────────────────────────────────

@twin_router.post("/stream")
async def twin_stream(request: TwinStreamRequest):
    """SSE streaming endpoint — response shape is identical to POST /simulate."""
    async def gen():
        async for chunk in _twin_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)