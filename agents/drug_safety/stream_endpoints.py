"""
agents/drug_safety/stream_endpoints.py — v3.0

Streaming pipeline mirrors main.py v3.0 exactly:
  Phase 0  — Lab context assessment (new status event)
  Phase 1  — Deterministic checks (+ lab-driven NSAID check)
  Phase 2  — FDA interactions + apply_severity_overrides (new status event)
  Phase 3  — Safety status
  Phase 4a — LLM interaction enrichment (stream tokens, with lab context)
  Phase 4b — LLM patient risk profile (stream tokens, with lab context)
  Phase 5a — LLM veto filter
  Phase 5b — Proactive alternatives for every flagged drug (stream tokens)
  Phase 5c — Drug-specific FHIR notes
  Final    — Persist + complete event
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

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

drug_router = APIRouter()


class DrugStreamRequest(BaseModel):
    proposed_medications: list[str]
    current_medications:  list[str] = []
    patient_allergies:    list[dict] = []
    active_conditions:    list[dict] = []
    patient_id:           str = "unknown"
    patient_state:        Optional[dict] = None
    enrich_with_llm:      bool = True


async def _drug_stream(req: DrugStreamRequest) -> AsyncIterator[str]:
    node  = "drug_safety"
    timer = Timer()
    request_id = str(uuid.uuid4())[:8]

    from safety_core import (
        check_allergy_cross_reactivity,
        check_condition_contraindications,
        build_fhir_medication_request,
        apply_severity_overrides,
        assess_critical_labs,
        generate_proactive_alternatives,
        InteractionEnrichmentBatch,
        PatientRiskProfile,
        AlternativeSuggestionsOutput,
        _INTERACTION_ENRICHMENT_PROMPT,
        _RISK_PROFILE_PROMPT,
        _PROACTIVE_ALTERNATIVES_PROMPT,
        ClinicalLabContext,
    )
    from fda_client import get_rxcuis_batch, check_drug_interactions_by_name, get_fda_warnings_batch
    from langchain_google_genai import ChatGoogleGenerativeAI
    from db import save_drug_safety, DrugSafetyRecord

    # ── Phase 0: Lab context ──────────────────────────────────────────────────
    yield evt_status(node, "Phase 0: Assessing critical lab values...", step=0, total=7)
    lab_results = (req.patient_state or {}).get("lab_results", [])
    lab_context: ClinicalLabContext = assess_critical_labs(lab_results)

    lab_summary_msg = lab_context.overall_lab_summary
    if lab_context.sepsis_suspicion:
        lab_summary_msg += " ⚠️ Sepsis suspected — antibiotic urgency elevated."
    yield evt_progress(node, lab_summary_msg, pct=8)

    # ── Phase 1: Deterministic checks ────────────────────────────────────────
    yield evt_status(
        node,
        f"Phase 1: Checking {len(req.proposed_medications)} proposed medications — "
        "allergy + condition + lab-driven contraindications...",
        step=1, total=7,
    )

    contraindications: list[dict] = []
    approved: list[str] = []
    flagged: list[str] = []
    drug_contraindications: dict[str, list[dict]] = {}

    for drug in req.proposed_medications:
        drug_contras: list[dict] = []
        drug_contras.extend(check_allergy_cross_reactivity(drug, req.patient_allergies))
        drug_contras.extend(check_condition_contraindications(drug, req.active_conditions))

        if lab_context.renal_impairment_suspected:
            drug_base = drug.lower().split()[0].strip().rstrip(".,")
            if drug_base in ["ibuprofen", "naproxen", "diclofenac", "ketorolac", "meloxicam"]:
                if not any("renal" in c.get("reason", "").lower() for c in drug_contras):
                    drug_contras.append({
                        "drug": drug,
                        "condition_code": "LAB",
                        "condition_display": "Suspected renal impairment (from lab values)",
                        "severity": "HIGH",
                        "reason": (
                            f"Lab values suggest renal impairment — NSAIDs worsen renal perfusion. "
                            f"{lab_context.overall_lab_summary}"
                        ),
                        "recommendation": f"Avoid {drug} given suspected renal impairment.",
                        "type": "lab_driven_contraindication",
                    })

        drug_contraindications[drug] = drug_contras
        if drug_contras:
            contraindications.extend(drug_contras)
            flagged.append(drug)
        else:
            approved.append(drug)

    yield evt_progress(
        node,
        f"Local checks: {len(approved)} approved, {len(flagged)} flagged, "
        f"{len(contraindications)} contraindications",
        pct=20,
    )

    # ── Phase 2: External APIs + severity overrides ───────────────────────────
    all_drugs = req.proposed_medications + req.current_medications
    yield evt_status(
        node,
        f"Phase 2: FDA interaction check across {len(all_drugs)} drugs + severity overrides...",
        step=2, total=7,
    )

    rxcui_map: dict = {}
    rxnav_interactions: list = []
    fda_warnings: dict = {}

    if len(all_drugs) >= 2:
        rxcui_map, fda_warnings = await asyncio.gather(
            get_rxcuis_batch(all_drugs),
            get_fda_warnings_batch(req.proposed_medications),
        )
        rxnav_interactions = await check_drug_interactions_by_name(all_drugs)
        if rxnav_interactions:
            rxnav_interactions = apply_severity_overrides(rxnav_interactions)
    else:
        fda_warnings = await get_fda_warnings_batch(req.proposed_medications)

    upgraded = [i for i in rxnav_interactions if i.get("severity_upgraded")]
    black_box_count = sum(1 for ws in fda_warnings.values() if any("[BLACK BOX]" in w for w in ws))

    yield evt_progress(
        node,
        f"{len(rxnav_interactions)} interaction(s) detected"
        + (f", {len(upgraded)} severity upgrade(s)" if upgraded else "")
        + f", {black_box_count} black box warning(s)",
        pct=35,
    )

    # ── Phase 3: Safety status ────────────────────────────────────────────────
    yield evt_status(node, "Phase 3: Determining safety status...", step=3, total=7)

    has_critical = any(c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications)
    has_severe_interaction = any(i.get("severity", "").upper() in ("HIGH", "CRITICAL") for i in rxnav_interactions)

    if has_critical or has_severe_interaction:
        safety_status = "UNSAFE"
    elif contraindications or rxnav_interactions or black_box_count > 0:
        safety_status = "CAUTION"
    else:
        safety_status = "SAFE"

    yield evt_progress(node, f"Safety verdict: {safety_status}", pct=42)

    # ── Phase 4: LLM enrichment (streamed tokens) ─────────────────────────────
    enriched_interactions_obj = None
    risk_profile_obj = None
    enriched_rxnav = list(rxnav_interactions)

    _patient_ctx = req.patient_state or {
        "demographics": {"age": "unknown", "gender": "unknown"},
        "active_conditions": req.active_conditions,
        "medications": [{"drug": m} for m in req.current_medications],
        "allergies": req.patient_allergies,
    }

    if req.enrich_with_llm:
        import json as _json
        demographics = _patient_ctx.get("demographics", {})
        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display','')})"
            for c in _patient_ctx.get("active_conditions", [])[:6]
        ) or "None documented"
        meds_str = ", ".join(m.get("drug","") for m in _patient_ctx.get("medications",[])[:6]) or "None documented"
        allergies_str = ", ".join(
            f"{a.get('substance')} → {a.get('reaction','?')} ({a.get('severity','?')})"
            for a in _patient_ctx.get("allergies",[])
        ) or "NKDA"
        lab_str = lab_context.overall_lab_summary
        if lab_context.drug_selection_constraints:
            lab_str += " Constraints: " + "; ".join(lab_context.drug_selection_constraints)

        # ── 4a: Interaction enrichment ────────────────────────────────────────
        if rxnav_interactions:
            yield evt_status(node, "Phase 4a: LLM enriching interactions (with lab context)...", step=4, total=7)

            llm_enrich = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
            structured_enrich = llm_enrich.with_structured_output(InteractionEnrichmentBatch)
            interactions_summary = [
                {"drug_a": i.get("drug_a"), "drug_b": i.get("drug_b"),
                 "severity": i.get("severity"), "rxnav_description": i.get("description",""),
                 "severity_upgraded": i.get("severity_upgraded", False),
                 "upgrade_reason": i.get("upgrade_reason","")}
                for i in rxnav_interactions
            ]
            enrich_chain = _INTERACTION_ENRICHMENT_PROMPT | structured_enrich
            enrich_tokens = 0

            try:
                async for event in enrich_chain.astream_events(
                    {"age": demographics.get("age","unknown"), "gender": demographics.get("gender","unknown"),
                     "conditions": conditions_str, "current_medications": meds_str,
                     "lab_context": lab_str,
                     "interactions_json": _json.dumps(interactions_summary, indent=2)},
                    version="v2",
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            enrich_tokens += 1
                            yield evt_token(node, chunk.content)
                    elif event["event"] == "on_chain_end" and event["name"] == "RunnableSequence":
                        raw = event["data"].get("output")
                        if isinstance(raw, InteractionEnrichmentBatch):
                            enriched_interactions_obj = raw
                        elif isinstance(raw, dict):
                            try:
                                enriched_interactions_obj = InteractionEnrichmentBatch(**raw)
                            except Exception as pe:
                                yield evt_error(node, f"InteractionEnrichmentBatch parse: {pe}", fatal=False)
            except Exception as exc:
                yield evt_error(node, f"Interaction enrichment streaming failed: {exc}", fatal=False)
                try:
                    enriched_interactions_obj = await enrich_chain.ainvoke(
                        {"age": demographics.get("age","unknown"), "gender": demographics.get("gender","unknown"),
                         "conditions": conditions_str, "current_medications": meds_str,
                         "lab_context": lab_str,
                         "interactions_json": _json.dumps(interactions_summary, indent=2)}
                    )
                except Exception as exc2:
                    yield evt_error(node, f"Interaction enrichment fallback failed: {exc2}", fatal=False)

            yield evt_progress(node, f"Interaction enrichment: {enrich_tokens} tokens", pct=55)

            if enriched_interactions_obj and enriched_interactions_obj.enriched_interactions:
                for i, e in enumerate(enriched_interactions_obj.enriched_interactions):
                    if i < len(enriched_rxnav):
                        enriched_rxnav[i] = {**enriched_rxnav[i],
                            "mechanism": e.mechanism, "clinical_significance": e.clinical_significance,
                            "monitoring_parameters": e.monitoring_parameters,
                            "management_strategy": e.management_strategy, "time_to_onset": e.time_to_onset}

        # ── 4b: Risk profile ──────────────────────────────────────────────────
        yield evt_status(node, "Phase 4b: LLM risk profile (with lab context)...", step=5, total=7)

        renal_conditions = [c for c in _patient_ctx.get("active_conditions",[])
                            if c.get("code","").startswith("N18") or c.get("code","").startswith("N17")]
        renal_hint = f"Renal impairment noted: {renal_conditions[0].get('display')}" if renal_conditions \
                     else "No renal impairment conditions documented"
        if lab_context.renal_impairment_suspected:
            renal_hint += " (also suspected from lab values)"

        llm_risk = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
        structured_risk = llm_risk.with_structured_output(PatientRiskProfile)
        risk_chain = _RISK_PROFILE_PROMPT | structured_risk
        risk_tokens = 0

        try:
            async for event in risk_chain.astream_events(
                {"age": demographics.get("age","unknown"), "gender": demographics.get("gender","unknown"),
                 "conditions": conditions_str, "current_medications": meds_str,
                 "allergies": allergies_str, "renal_hint": renal_hint,
                 "lab_context": lab_str,
                 "proposed_meds": ", ".join(req.proposed_medications),
                 "contra_count": len(contraindications),
                 "contraindications_json": _json.dumps(
                     [{"drug":c["drug"],"severity":c["severity"],"reason":c["reason"]} for c in contraindications[:5]], indent=2),
                 "interaction_count": len(rxnav_interactions),
                 "interactions_json": _json.dumps(
                     [{"drugs":f"{i.get('drug_a')}+{i.get('drug_b')}","severity":i.get("severity")} for i in rxnav_interactions[:5]], indent=2),
                 "fda_warnings_json": _json.dumps({k:v[:2] for k,v in list(fda_warnings.items())[:3] if v}),
                 "safety_status": safety_status},
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        risk_tokens += 1
                        yield evt_token(node, chunk.content)
                elif event["event"] == "on_chain_end" and event["name"] == "RunnableSequence":
                    raw = event["data"].get("output")
                    if isinstance(raw, PatientRiskProfile):
                        risk_profile_obj = raw
                    elif isinstance(raw, dict):
                        try:
                            risk_profile_obj = PatientRiskProfile(**raw)
                        except Exception as pe:
                            yield evt_error(node, f"PatientRiskProfile parse: {pe}", fatal=False)
        except Exception as exc:
            yield evt_error(node, f"Risk profile streaming failed: {exc}", fatal=False)

        yield evt_progress(
            node,
            f"Risk profile: {risk_tokens} tokens — "
            f"{risk_profile_obj.overall_risk_level if risk_profile_obj else 'N/A'}",
            pct=70,
        )

    # ── Phase 5a: LLM veto ────────────────────────────────────────────────────
    llm_vetoed = False
    if risk_profile_obj:
        if not risk_profile_obj.safe_to_proceed and risk_profile_obj.overall_risk_level in ("CRITICAL", "HIGH"):
            llm_vetoed = True
            yield evt_progress(
                node,
                f"LLM veto: safe_to_proceed=False at {risk_profile_obj.overall_risk_level} — "
                f"moving {len(approved)} approved drug(s) to flagged",
                pct=73,
            )
            for drug in approved:
                entry = {
                    "drug": drug, "allergen": None,
                    "reason": f"LLM risk veto: {risk_profile_obj.overall_risk_level} risk. {risk_profile_obj.clinical_summary}",
                    "severity": risk_profile_obj.overall_risk_level,
                    "recommendation": risk_profile_obj.recommended_action,
                    "type": "llm_risk_veto",
                }
                contraindications.append(entry)
                drug_contraindications.setdefault(drug, []).append(entry)
            flagged = flagged + approved
            approved = []
            safety_status = "UNSAFE"

    # ── Phase 5b: Proactive alternatives (streamed) ───────────────────────────
    proactive_alternatives: dict[str, Optional[dict]] = {}
    if req.enrich_with_llm and flagged:
        yield evt_status(
            node,
            f"Phase 5b: Generating alternatives for {len(flagged)} flagged drug(s)...",
            step=6, total=7,
        )

        alt_results = await asyncio.gather(*[
            generate_proactive_alternatives(
                drug_name=drug,
                contraindications_for_drug=drug_contraindications.get(drug, []),
                patient_state=_patient_ctx,
                lab_context=lab_context,
                active_conditions=req.active_conditions,
                current_medications=[{"drug": m} for m in req.current_medications],
                allergies=req.patient_allergies,
            )
            for drug in flagged
        ])

        for drug, result in zip(flagged, alt_results):
            proactive_alternatives[drug] = result.model_dump() if result else None
            if result:
                alt_names = [a.drug for a in result.alternatives]
                yield evt_progress(node, f"Alternatives for {drug}: {', '.join(alt_names)}", pct=85)

    # ── Phase 5c: Drug-specific FHIR notes ───────────────────────────────────
    risk_level = risk_profile_obj.overall_risk_level if risk_profile_obj else "LOW"
    lab_note = lab_context.overall_lab_summary if lab_context.critical_flags else ""

    def _get_specific_reason(drug: str) -> str:
        drug_contras = drug_contraindications.get(drug, [])
        if not drug_contras:
            return "Safety concern identified."
        top = max(drug_contras,
                  key=lambda c: {"CRITICAL":3,"HIGH":2,"MODERATE":1,"LOW":0}.get(c.get("severity","LOW"),0))
        reason = top.get("reason", "")
        drug_base = drug.lower().split()[0]
        for i in enriched_rxnav:
            if drug_base in i.get("drug_a","").lower() or drug_base in i.get("drug_b","").lower():
                other = i.get("drug_b") if drug_base in i.get("drug_a","").lower() else i.get("drug_a")
                reason += f" Additionally: {i.get('severity')} interaction with {other}."
                break
        return reason

    fhir_meds = []
    for drug in approved:
        interaction_note = next(
            (f"{i.get('severity')} interaction with "
             f"{i.get('drug_b') if drug.lower().split()[0] in i.get('drug_a','').lower() else i.get('drug_a')} — monitor."
             for i in enriched_rxnav
             if drug.lower().split()[0] in i.get("drug_a","").lower()
             or drug.lower().split()[0] in i.get("drug_b","").lower()),
            ""
        )
        fhir_meds.append(build_fhir_medication_request(
            drug, req.patient_id, safety_cleared=True,
            safety_note=interaction_note, risk_level=risk_level, lab_context_note=lab_note))
    for drug in flagged:
        fhir_meds.append(build_fhir_medication_request(
            drug, req.patient_id, safety_cleared=False,
            risk_level=risk_level, specific_reason=_get_specific_reason(drug), lab_context_note=lab_note))

    patient_risk_profile_dict = risk_profile_obj.model_dump() if risk_profile_obj else None
    interaction_risk_narrative = (
        enriched_interactions_obj.overall_risk_narrative if enriched_interactions_obj else None
    )

    result = {
        "safety_status":  safety_status,
        "lab_assessment": {
            "critical_flags":               [f.model_dump() for f in lab_context.critical_flags],
            "sepsis_suspicion":             lab_context.sepsis_suspicion,
            "renal_impairment_suspected":   lab_context.renal_impairment_suspected,
            "hepatic_impairment_suspected": lab_context.hepatic_impairment_suspected,
            "coagulopathy_suspected":       lab_context.coagulopathy_suspected,
            "overall_lab_summary":          lab_context.overall_lab_summary,
            "drug_selection_constraints":   lab_context.drug_selection_constraints,
        },
        "contraindications":        contraindications,
        "critical_interactions":    enriched_rxnav,
        "fda_warnings":             fda_warnings,
        "rxcui_map":                rxcui_map,
        "approved_medications":     approved,
        "flagged_medications":      flagged,
        "fhir_medication_requests": fhir_meds,
        "proactive_alternatives":   proactive_alternatives,
        "patient_risk_profile":     patient_risk_profile_dict,
        "interaction_risk_narrative": interaction_risk_narrative,
        "summary": {
            "proposed_count":         len(req.proposed_medications),
            "approved_count":         len(approved),
            "flagged_count":          len(flagged),
            "interaction_count":      len(rxnav_interactions),
            "contraindication_count": len(contraindications),
            "black_box_warnings":     black_box_count,
            "severity_upgrades":      len(upgraded),
            "critical_lab_flags":     len(lab_context.critical_flags),
            "sepsis_suspicion":       lab_context.sepsis_suspicion,
            "alternatives_generated": len([v for v in proactive_alternatives.values() if v]),
            "llm_enriched":           req.enrich_with_llm and risk_profile_obj is not None,
            "llm_vetoed":             llm_vetoed,
        },
        "request_id": request_id,
        "cache_hit":  False,
    }

    await save_drug_safety(DrugSafetyRecord(
        request_id=request_id, patient_id=req.patient_id,
        proposed_medications=req.proposed_medications, current_medications=req.current_medications,
        safety_status=safety_status, contraindications=contraindications,
        approved_medications=approved, flagged_medications=flagged,
        critical_interactions=enriched_rxnav, fda_warnings=fda_warnings,
        patient_risk_profile=patient_risk_profile_dict,
        interaction_risk_narrative=interaction_risk_narrative,
        fhir_medication_requests=fhir_meds,
        llm_enriched=req.enrich_with_llm and risk_profile_obj is not None,
        cache_hit=False, elapsed_ms=timer.elapsed_ms(), source="stream",
    ))

    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())


@drug_router.post("/stream")
async def drug_stream(request: DrugStreamRequest):
    """SSE streaming drug safety check — v3.0."""
    async def gen():
        async for chunk in _drug_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)