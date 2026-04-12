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
# DRUG SAFETY  (/stream on port 8004)
# ════════════════════════════════════════════════════════════════════════════════
 
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
 
    yield evt_status(node,
                     f"Checking {len(req.proposed_medications)} proposed medications...",
                     step=1, total=4)
 
    from safety_core import (
        check_allergy_cross_reactivity,
        check_condition_contraindications,
        build_fhir_medication_request,
        enrich_interactions_with_llm,
        generate_patient_risk_profile,
    )
    from fda_client import get_rxcuis_batch, check_drug_interactions_by_name, get_fda_warnings_batch
 
    # Step 1 — Deterministic local checks
    yield evt_status(node, "Phase 1: Allergy cross-reactivity + condition contraindications...",
                     step=1, total=4)
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
 
    yield evt_progress(node,
                       f"Local check: {len(approved)} approved, {len(flagged)} flagged",
                       pct=25)
 
    # Step 2 — External APIs (concurrent)
    yield evt_status(node, "Phase 2: Fetching RxNav interactions + FDA warnings...",
                     step=2, total=4)
    all_drugs = req.proposed_medications + req.current_medications
    rxcui_map, fda_warnings = await asyncio.gather(
        get_rxcuis_batch(all_drugs),
        get_fda_warnings_batch(req.proposed_medications),
    )
    rxcuis = [v for v in rxcui_map.values() if v]
    rxnav_interactions = []
    if len(rxcuis) >= 2:
        rxnav_interactions = await check_drug_interactions_by_name(all_drugs)
 
    yield evt_progress(node,
                       f"External: {len(rxnav_interactions)} interactions, "
                       f"{sum(1 for w in fda_warnings.values() if w)} FDA warnings",
                       pct=55)
 
    # Step 3 — Safety status
    has_critical = any(c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications)
    has_severe   = any(i.get("severity", "").upper() in ("HIGH", "CRITICAL") for i in rxnav_interactions)
    has_bb       = any(any("[BLACK BOX]" in w for w in ws) for ws in fda_warnings.values() if ws)
 
    safety_status = "UNSAFE" if (has_critical or has_severe) else ("CAUTION" if (contraindications or rxnav_interactions or has_bb) else "SAFE")
 
    # Step 4 — LLM enrichment
    enriched_interactions, patient_risk_profile = None, None
    if req.enrich_with_llm and rxnav_interactions:
        yield evt_status(node, "Phase 3: LLM interaction enrichment + risk profile...",
                         step=3, total=4)
        _ctx = req.patient_state or {
            "demographics": {"age": "unknown", "gender": "unknown"},
            "active_conditions": req.active_conditions,
            "medications": [{"drug": m} for m in req.current_medications],
            "allergies": req.patient_allergies,
        }
        ie, rp = await asyncio.gather(
            enrich_interactions_with_llm(rxnav_interactions, _ctx),
            generate_patient_risk_profile(
                patient_state=_ctx,
                proposed_medications=req.proposed_medications,
                contraindications=contraindications,
                interactions=rxnav_interactions,
                fda_warnings=fda_warnings,
                safety_status=safety_status,
            ),
        )
        if ie:
            enriched_interactions = ie.model_dump()
        if rp:
            patient_risk_profile = rp.model_dump()
 
    yield evt_progress(node, f"Safety status: {safety_status}", pct=90)
 
    # FHIR MedicationRequests
    risk_level = patient_risk_profile.get("overall_risk_level", "LOW") if patient_risk_profile else "LOW"
    fhir_meds  = [
        build_fhir_medication_request(drug, req.patient_id, safety_cleared=True,
                                      risk_level=risk_level)
        for drug in approved
    ]
 
    result = {
        "safety_status":          safety_status,
        "contraindications":      contraindications,
        "critical_interactions":  rxnav_interactions,
        "fda_warnings":           fda_warnings,
        "approved_medications":   approved,
        "flagged_medications":    flagged,
        "fhir_medication_requests": fhir_meds,
        "patient_risk_profile":   patient_risk_profile,
        "alternatives":           [],
        "summary": {
            "approved_count":        len(approved),
            "flagged_count":         len(flagged),
            "interaction_count":     len(rxnav_interactions),
            "contraindication_count": len(contraindications),
            "black_box_warnings":    sum(1 for w in fda_warnings.values() if any("[BLACK BOX]" in x for x in w)),
            "llm_enriched":          req.enrich_with_llm and patient_risk_profile is not None,
        },
    }
 
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@drug_router.post("/stream")
async def drug_stream(request: DrugStreamRequest):
    async def gen():
        async for chunk in _drug_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
 