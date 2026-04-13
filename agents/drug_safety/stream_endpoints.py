"""
agents/drug_safety/stream_endpoints.py
----------------------------------------
SSE streaming endpoint for the Drug Safety Agent.

Mirrors diagnosis/stream_endpoint.py pattern:
  - astream_events(version="v2") for live LLM token streaming
  - on_chat_model_stream  → yield evt_token (enrichment tokens live to client)
  - on_chain_end / RunnableSequence → capture structured Pydantic output
  - DB persistence via save_drug_safety (same as /check-safety)
  - Response shape matches /check-safety exactly

Pipeline (streamed with progress events):
  Step 1 — Deterministic local checks (allergy cross-reactivity + condition contraindications)
  Step 2 — External APIs concurrent (RxNav interactions + FDA warnings)
  Step 3 — Safety status determination
  Step 4 — LLM interaction enrichment  ← streamed tokens live
  Step 5 — LLM patient risk profile    ← streamed tokens live
  Step 6 — Assemble + persist
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
        InteractionEnrichmentBatch,
        PatientRiskProfile,
        _INTERACTION_ENRICHMENT_PROMPT,
        _RISK_PROFILE_PROMPT,
    )
    from fda_client import (
        get_rxcuis_batch,
        check_drug_interactions_by_name,
        get_fda_warnings_batch,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from db import save_drug_safety, DrugSafetyRecord

    # ── Step 1: Deterministic local checks ────────────────────────────────────
    yield evt_status(
        node,
        f"Phase 1: Checking {len(req.proposed_medications)} proposed medications — "
        "allergy cross-reactivity + condition contraindications...",
        step=1, total=5,
    )

    contraindications, approved, flagged = [], [], []
    for drug in req.proposed_medications:
        drug_contras = []
        drug_contras.extend(check_allergy_cross_reactivity(drug, req.patient_allergies))
        drug_contras.extend(check_condition_contraindications(drug, req.active_conditions))
        if drug_contras:
            contraindications.extend(drug_contras)
            flagged.append(drug)
        else:
            approved.append(drug)

    yield evt_progress(
        node,
        f"Local check done: {len(approved)} approved, {len(flagged)} flagged, "
        f"{len(contraindications)} contraindications",
        pct=20,
    )

    # ── Step 2: External APIs (concurrent) ────────────────────────────────────
    yield evt_status(
        node, "Phase 2: Fetching RxNav interactions + FDA warnings (concurrent)...",
        step=2, total=5,
    )

    all_drugs = req.proposed_medications + req.current_medications
    rxcui_map, fda_warnings = await asyncio.gather(
        get_rxcuis_batch(all_drugs),
        get_fda_warnings_batch(req.proposed_medications),
    )

    rxcuis = [v for v in rxcui_map.values() if v]
    rxnav_interactions: list[dict] = []
    if len(rxcuis) >= 2:
        rxnav_interactions = await check_drug_interactions_by_name(all_drugs)

    black_box_count = sum(
        1 for ws in fda_warnings.values()
        if any("[BLACK BOX]" in w for w in ws)
    )

    yield evt_progress(
        node,
        f"External APIs: {len(rxnav_interactions)} interactions detected, "
        f"{black_box_count} black box warning(s)",
        pct=40,
    )

    # ── Step 3: Safety status ─────────────────────────────────────────────────
    yield evt_status(node, "Phase 3: Determining safety status...", step=3, total=5)

    has_critical = any(
        c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications
    )
    has_severe_interaction = any(
        i.get("severity", "").upper() in ("HIGH", "CRITICAL")
        for i in rxnav_interactions
    )
    has_black_box = black_box_count > 0

    if has_critical or has_severe_interaction:
        safety_status = "UNSAFE"
    elif contraindications or rxnav_interactions or has_black_box:
        safety_status = "CAUTION"
    else:
        safety_status = "SAFE"

    yield evt_progress(node, f"Safety verdict: {safety_status}", pct=50)

    # ── Step 4 & 5: LLM enrichment with token streaming ──────────────────────
    enriched_interactions_obj = None
    risk_profile_obj = None
    enriched_rxnav = list(rxnav_interactions)

    if req.enrich_with_llm:
        _patient_ctx = req.patient_state or {
            "demographics": {"age": "unknown", "gender": "unknown"},
            "active_conditions": req.active_conditions,
            "medications": [{"drug": m} for m in req.current_medications],
            "allergies": req.patient_allergies,
        }

        # ── 4a: Interaction enrichment (stream tokens) ────────────────────────
        if rxnav_interactions:
            yield evt_status(
                node,
                "Phase 4a: LLM enriching interactions (mechanism, monitoring, management)...",
                step=4, total=5,
            )

            import json as _json
            llm_enrich = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", temperature=0.0
            )
            structured_enrich = llm_enrich.with_structured_output(InteractionEnrichmentBatch)

            demographics = _patient_ctx.get("demographics", {})
            conditions_str = ", ".join(
                f"{c.get('code')} ({c.get('display', '')})"
                for c in _patient_ctx.get("active_conditions", [])[:6]
            ) or "None documented"
            meds_str = ", ".join(
                m.get("drug", "") for m in _patient_ctx.get("medications", [])[:6]
            ) or "None documented"

            interactions_summary = [
                {
                    "drug_a": i.get("drug_a"),
                    "drug_b": i.get("drug_b"),
                    "severity": i.get("severity"),
                    "rxnav_description": i.get("description", ""),
                }
                for i in rxnav_interactions
            ]

            enrich_chain = _INTERACTION_ENRICHMENT_PROMPT | structured_enrich
            enrich_token_count = 0

            try:
                async for event in enrich_chain.astream_events(
                    {
                        "age": demographics.get("age", "unknown"),
                        "gender": demographics.get("gender", "unknown"),
                        "conditions": conditions_str,
                        "current_medications": meds_str,
                        "interactions_json": _json.dumps(interactions_summary, indent=2),
                    },
                    version="v2",
                ):
                    kind = event["event"]
                    name = event["name"]

                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            enrich_token_count += 1
                            yield evt_token(node, chunk.content)

                    elif kind == "on_chain_end" and name == "RunnableSequence":
                        raw = event["data"].get("output")
                        if isinstance(raw, InteractionEnrichmentBatch):
                            enriched_interactions_obj = raw
                        elif isinstance(raw, dict):
                            try:
                                enriched_interactions_obj = InteractionEnrichmentBatch(**raw)
                            except Exception as pe:
                                yield evt_error(
                                    node,
                                    f"InteractionEnrichmentBatch parse failed: {pe}",
                                    fatal=False,
                                )

            except Exception as exc:
                yield evt_error(
                    node,
                    f"Interaction enrichment streaming failed: {exc} — attempting sync fallback",
                    fatal=False,
                )
                try:
                    enriched_interactions_obj = await enrich_chain.ainvoke(
                        {
                            "age": demographics.get("age", "unknown"),
                            "gender": demographics.get("gender", "unknown"),
                            "conditions": conditions_str,
                            "current_medications": meds_str,
                            "interactions_json": _json.dumps(interactions_summary, indent=2),
                        }
                    )
                except Exception as exc2:
                    yield evt_error(
                        node,
                        f"Interaction enrichment completely failed: {exc2}",
                        fatal=False,
                    )

            yield evt_progress(
                node,
                f"Interaction enrichment: {enrich_token_count} tokens streamed",
                pct=65,
            )

            # Merge enrichment into rxnav list
            if enriched_interactions_obj and enriched_interactions_obj.enriched_interactions:
                for i, enrichment in enumerate(
                    enriched_interactions_obj.enriched_interactions
                ):
                    if i < len(enriched_rxnav):
                        enriched_rxnav[i] = {
                            **enriched_rxnav[i],
                            "mechanism":             enrichment.mechanism,
                            "clinical_significance": enrichment.clinical_significance,
                            "monitoring_parameters": enrichment.monitoring_parameters,
                            "management_strategy":   enrichment.management_strategy,
                            "time_to_onset":         enrichment.time_to_onset,
                        }

        # ── 4b: Patient risk profile (stream tokens) ──────────────────────────
        yield evt_status(
            node,
            "Phase 4b: LLM generating patient-specific risk profile...",
            step=5, total=5,
        )

        llm_risk = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", temperature=0.0
        )
        structured_risk = llm_risk.with_structured_output(PatientRiskProfile)

        demographics = _patient_ctx.get("demographics", {})
        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in _patient_ctx.get("active_conditions", [])[:6]
        ) or "None"
        meds_str = ", ".join(
            m.get("drug", "") for m in _patient_ctx.get("medications", [])[:6]
        ) or "None"
        allergies_str = ", ".join(
            f"{a.get('substance')} → {a.get('reaction', '?')} ({a.get('severity', '?')})"
            for a in _patient_ctx.get("allergies", [])
        ) or "NKDA"

        renal_conditions = [
            c for c in _patient_ctx.get("active_conditions", [])
            if c.get("code", "").startswith("N18") or c.get("code", "").startswith("N17")
        ]
        renal_hint = (
            f"Renal impairment noted: {renal_conditions[0].get('display')}"
            if renal_conditions
            else "No renal impairment conditions documented"
        )

        import json as _json2
        risk_chain = _RISK_PROFILE_PROMPT | structured_risk
        risk_token_count = 0

        try:
            async for event in risk_chain.astream_events(
                {
                    "age":                demographics.get("age", "unknown"),
                    "gender":             demographics.get("gender", "unknown"),
                    "conditions":         conditions_str,
                    "current_medications": meds_str,
                    "allergies":          allergies_str,
                    "renal_hint":         renal_hint,
                    "proposed_meds":      ", ".join(req.proposed_medications),
                    "contra_count":       len(contraindications),
                    "contraindications_json": _json2.dumps(
                        [
                            {
                                "drug": c["drug"],
                                "severity": c["severity"],
                                "reason": c["reason"],
                            }
                            for c in contraindications[:5]
                        ],
                        indent=2,
                    ),
                    "interaction_count":  len(rxnav_interactions),
                    "interactions_json":  _json2.dumps(
                        [
                            {
                                "drugs": f"{i.get('drug_a')} + {i.get('drug_b')}",
                                "severity": i.get("severity"),
                            }
                            for i in rxnav_interactions[:5]
                        ],
                        indent=2,
                    ),
                    "fda_warnings_json":  _json2.dumps(
                        {
                            k: v[:2]
                            for k, v in list(fda_warnings.items())[:3]
                            if v
                        }
                    ),
                    "safety_status":      safety_status,
                },
                version="v2",
            ):
                kind = event["event"]
                name = event["name"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        risk_token_count += 1
                        yield evt_token(node, chunk.content)

                elif kind == "on_chain_end" and name == "RunnableSequence":
                    raw = event["data"].get("output")
                    if isinstance(raw, PatientRiskProfile):
                        risk_profile_obj = raw
                    elif isinstance(raw, dict):
                        try:
                            risk_profile_obj = PatientRiskProfile(**raw)
                        except Exception as pe:
                            yield evt_error(
                                node,
                                f"PatientRiskProfile parse failed: {pe}",
                                fatal=False,
                            )

        except Exception as exc:
            yield evt_error(
                node,
                f"Risk profile streaming failed: {exc} — attempting sync fallback",
                fatal=False,
            )
            try:
                risk_profile_obj = await risk_chain.ainvoke(
                    {
                        "age":                demographics.get("age", "unknown"),
                        "gender":             demographics.get("gender", "unknown"),
                        "conditions":         conditions_str,
                        "current_medications": meds_str,
                        "allergies":          allergies_str,
                        "renal_hint":         renal_hint,
                        "proposed_meds":      ", ".join(req.proposed_medications),
                        "contra_count":       len(contraindications),
                        "contraindications_json": "[]",
                        "interaction_count":  len(rxnav_interactions),
                        "interactions_json":  "[]",
                        "fda_warnings_json":  "{}",
                        "safety_status":      safety_status,
                    }
                )
            except Exception as exc2:
                yield evt_error(
                    node,
                    f"Risk profile completely failed: {exc2}",
                    fatal=False,
                )

        yield evt_progress(
            node,
            f"Risk profile: {risk_token_count} tokens streamed — "
            f"risk level: {risk_profile_obj.overall_risk_level if risk_profile_obj else 'N/A'}",
            pct=88,
        )

    # ── Step 6: Assemble FHIR + result ────────────────────────────────────────
    risk_level = (
        risk_profile_obj.overall_risk_level
        if risk_profile_obj else "LOW"
    )
    fhir_meds = [
        build_fhir_medication_request(
            drug, req.patient_id,
            safety_cleared=True,
            risk_level=risk_level,
        )
        for drug in approved
    ]

    patient_risk_profile_dict = risk_profile_obj.model_dump() if risk_profile_obj else None
    interaction_risk_narrative = (
        enriched_interactions_obj.overall_risk_narrative
        if enriched_interactions_obj else None
    )

    result = {
        "safety_status":            safety_status,
        "contraindications":        contraindications,
        "critical_interactions":    enriched_rxnav,
        "fda_warnings":             fda_warnings,
        "rxcui_map":                rxcui_map,
        "approved_medications":     approved,
        "flagged_medications":      flagged,
        "fhir_medication_requests": fhir_meds,
        "patient_risk_profile":     patient_risk_profile_dict,
        "interaction_risk_narrative": interaction_risk_narrative,
        "summary": {
            "proposed_count":         len(req.proposed_medications),
            "approved_count":         len(approved),
            "flagged_count":          len(flagged),
            "interaction_count":      len(rxnav_interactions),
            "contraindication_count": len(contraindications),
            "black_box_warnings":     black_box_count,
            "llm_enriched":           req.enrich_with_llm and risk_profile_obj is not None,
        },
        "request_id": request_id,
        "cache_hit":  False,
    }

    # ── Persist to DB (non-fatal) ─────────────────────────────────────────────
    await save_drug_safety(
        DrugSafetyRecord(
            request_id=request_id,
            patient_id=req.patient_id,
            proposed_medications=req.proposed_medications,
            current_medications=req.current_medications,
            safety_status=safety_status,
            contraindications=contraindications,
            approved_medications=approved,
            flagged_medications=flagged,
            critical_interactions=enriched_rxnav,
            fda_warnings=fda_warnings,
            patient_risk_profile=patient_risk_profile_dict,
            interaction_risk_narrative=interaction_risk_narrative,
            fhir_medication_requests=fhir_meds,
            llm_enriched=req.enrich_with_llm and risk_profile_obj is not None,
            cache_hit=False,
            elapsed_ms=timer.elapsed_ms(),
            source="stream",
        )
    )

    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())


@drug_router.post("/stream")
async def drug_stream(request: DrugStreamRequest):
    """
    SSE streaming version of /check-safety.

    Streams:
      - Status/progress events for each pipeline phase
      - Live LLM tokens from interaction enrichment (Phase 4a)
      - Live LLM tokens from patient risk profile (Phase 4b)
      - Final complete event with full result dict
    """
    async def gen():
        async for chunk in _drug_stream(request):
            yield chunk
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)