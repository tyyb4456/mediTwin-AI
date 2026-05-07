"""
agents/drug_safety/main.py — v3.0 (Clinical Standards Upgrade)

New in v3.0 over v2.1.1:
─────────────────────────────────────────────────────────────────────────────
1. Phase 0: Lab context assessment (deterministic, before everything)
   - assess_critical_labs() reads patient_state.lab_results
   - Flags WBC, creatinine, INR, K+, ALT, Hgb, platelets, eGFR
   - sepsis_suspicion, renal_impairment_suspected, drug_selection_constraints
   - Injected into all LLM prompts and FHIR notes

2. Phase 2: Severity overrides after FDA label check
   - apply_severity_overrides() upgrades Warfarin+NSAID → HIGH, etc.
   - 20+ known dangerous pairs per ASHP/AHA/BNF guidelines
   - Never downgrades — only upgrades

3. Phase 5b: Proactive alternatives for every flagged drug
   - generate_proactive_alternatives() called concurrently for all flagged drugs
   - Lab context steers away from renally-cleared drugs if Cr is high
   - Sepsis flag tells LLM to prioritise broad-spectrum coverage
   - Result in proactive_alternatives dict keyed by drug name

4. Drug-specific FHIR notes
   - build_fhir_medication_request() receives specific_reason per drug
   - Amoxicillin: "Penicillin allergy — anaphylaxis cross-reactivity"
   - Ibuprofen: "Atrial fibrillation + Warfarin — NSAID bleeding risk"

5. lab_assessment field in API response
   - Full ClinicalLabContext in output for orchestrator + UI
"""
import os
import sys
import json
import uuid
import asyncio
from typing import Optional, Annotated
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from stream_endpoints import drug_router as stream_router

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fda_client import get_rxcuis_batch, check_drug_interactions_by_name, get_fda_warnings_batch
from safety_core import (
    check_allergy_cross_reactivity,
    check_condition_contraindications,
    get_alternative_suggestions_async,
    enrich_interactions_with_llm,
    generate_patient_risk_profile,
    generate_proactive_alternatives,
    build_fhir_medication_request,
    apply_severity_overrides,
    assess_critical_labs,
    ClinicalLabContext,
)

import db
from db import init as db_init, close as db_close, save_drug_safety, DrugSafetyRecord
from history_router import router as history_router

import logging
logger = logging.getLogger("drug_safety.main")

# ── Redis cache ───────────────────────────────────────────────────────────────

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            _redis = await aioredis.from_url(
                f"redis://{host}:{port}", encoding="utf-8", decode_responses=True
            )
            await _redis.ping()
        except Exception as e:
            logger.warning(f"  ⚠   Redis unavailable ({e}) — caching disabled")
            _redis = None
    return _redis


async def cache_get(key: str) -> Optional[dict]:
    r = await get_redis()
    if not r:
        return None
    try:
        val = await r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def cache_set(key: str, value: dict, ttl: int = 3600):
    r = await get_redis()
    if not r:
        return
    try:
        await r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Core safety pipeline — v3.0
# ══════════════════════════════════════════════════════════════════════════════

async def _run_full_safety_check(
    proposed_medications: list[str],
    current_medications: list[str],
    patient_allergies: list[dict],
    active_conditions: list[dict],
    patient_id: str = "unknown",
    patient_state: Optional[dict] = None,
    enrich_with_llm: bool = True,
    persist: bool = True,
    source: str = "check-safety",
) -> dict:
    all_drugs = proposed_medications + current_medications
    cache_key = (
        f"drug_safety_v3:"
        f"{':'.join(sorted(all_drugs))}:"
        f"{':'.join(sorted(a.get('substance','') for a in patient_allergies))}"
    )
    base_cached = await cache_get(cache_key)
    if base_cached and not enrich_with_llm:
        print("  ✓ Full cache hit")
        return base_cached

    # ── Phase 0: Lab context ──────────────────────────────────────────────────
    print("  Phase 0: Assessing critical lab values...")
    lab_results = (patient_state or {}).get("lab_results", [])
    lab_context: ClinicalLabContext = assess_critical_labs(lab_results)
    if lab_context.critical_flags:
        logger.warning(f"  Lab flags: {[f.display + '=' + str(f.value) + ' (' + f.flag + ')' for f in lab_context.critical_flags]}")
    if lab_context.sepsis_suspicion:
        logger.warning("  ⚠   Sepsis suspected — antibiotic urgency elevated")

    # ── Phase 1: Deterministic checks ────────────────────────────────────────
    contraindications: list[dict] = []
    approved: list[str] = []
    flagged: list[str] = []
    drug_contraindications: dict[str, list[dict]] = {}

    for drug in proposed_medications:
        drug_contras: list[dict] = []
        drug_contras.extend(check_allergy_cross_reactivity(drug, patient_allergies))
        drug_contras.extend(check_condition_contraindications(drug, active_conditions))

        # Lab-driven NSAID check when renal impairment suspected from labs
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
                        "recommendation": f"Avoid {drug} given suspected renal impairment. Use acetaminophen for analgesia.",
                        "type": "lab_driven_contraindication",
                    })

        drug_contraindications[drug] = drug_contras
        if drug_contras:
            contraindications.extend(drug_contras)
            flagged.append(drug)
        else:
            approved.append(drug)

    # ── Phase 2: External APIs + severity overrides ───────────────────────────
    rxcui_map: dict = {}
    rxnav_interactions: list = []
    fda_warnings: dict = {}

    if len(all_drugs) >= 2:
        rxcui_map, fda_warnings = await asyncio.gather(
            get_rxcuis_batch(all_drugs),
            get_fda_warnings_batch(proposed_medications),
        )
        rxnav_interactions = await check_drug_interactions_by_name(all_drugs)
        if rxnav_interactions:
            rxnav_interactions = apply_severity_overrides(rxnav_interactions)
            upgraded = [i for i in rxnav_interactions if i.get("severity_upgraded")]
            if upgraded:
                print(f"  Severity upgraded: {[(i['drug_a'],i['drug_b'],i['severity']) for i in upgraded]}")
    else:
        fda_warnings = await get_fda_warnings_batch(proposed_medications)

    logger.info(f"  {len(rxnav_interactions)} interaction(s) found.")

    # ── Phase 3: Safety status ────────────────────────────────────────────────
    has_critical_contra = any(c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications)
    has_severe_interaction = any(i.get("severity", "").upper() in ("HIGH", "CRITICAL") for i in rxnav_interactions)
    has_black_box = any(
        any("[BLACK BOX]" in w for w in ws) for ws in fda_warnings.values() if ws
    )

    if has_critical_contra or has_severe_interaction:
        safety_status = "UNSAFE"
    elif contraindications or rxnav_interactions or has_black_box:
        safety_status = "CAUTION"
    else:
        safety_status = "SAFE"

    # ── Phase 4: LLM enrichment ───────────────────────────────────────────────
    interaction_enrichment = None
    risk_profile = None
    patient_risk_profile = None

    _patient_ctx = patient_state or {
        "demographics": {"age": "unknown", "gender": "unknown"},
        "active_conditions": active_conditions,
        "medications": [{"drug": m} for m in current_medications],
        "allergies": patient_allergies,
    }

    if enrich_with_llm:
        print("  Phase 4: LLM enrichment (with lab context)...")
        interaction_enrichment, risk_profile = await asyncio.gather(
            enrich_interactions_with_llm(rxnav_interactions, _patient_ctx, lab_context),
            generate_patient_risk_profile(
                patient_state=_patient_ctx,
                proposed_medications=proposed_medications,
                contraindications=contraindications,
                interactions=rxnav_interactions,
                fda_warnings=fda_warnings,
                safety_status=safety_status,
                lab_context=lab_context,
            ),
        )
        if risk_profile:
            patient_risk_profile = risk_profile.model_dump()

    # ── Phase 5a: LLM veto ────────────────────────────────────────────────────
    llm_vetoed = False
    if patient_risk_profile:
        if not patient_risk_profile.get("safe_to_proceed", True) and \
           patient_risk_profile.get("overall_risk_level") in ("CRITICAL", "HIGH"):
            llm_vetoed = True
            logger,Warning(f"  ⚠   LLM veto: {patient_risk_profile.get('overall_risk_level')} risk, moving {len(approved)} approved to flagged")
            for drug in approved:
                entry = {
                    "drug": drug,
                    "allergen": None,
                    "reason": (
                        f"LLM risk veto: overall risk {patient_risk_profile.get('overall_risk_level')}, "
                        f"safe_to_proceed=False. {patient_risk_profile.get('clinical_summary','')}"
                    ),
                    "severity": patient_risk_profile.get("overall_risk_level"),
                    "recommendation": patient_risk_profile.get("recommended_action", "DO_NOT_PRESCRIBE"),
                    "type": "llm_risk_veto",
                }
                contraindications.append(entry)
                drug_contraindications.setdefault(drug, []).append(entry)
            flagged = flagged + approved
            approved = []
            safety_status = "UNSAFE"

    # ── Phase 5b: Proactive alternatives ─────────────────────────────────────
    proactive_alternatives: dict[str, Optional[dict]] = {}
    if enrich_with_llm and flagged:
        print(f"  Phase 5b: Proactive alternatives for {len(flagged)} flagged drug(s)...")
        alt_results = await asyncio.gather(*[
            generate_proactive_alternatives(
                drug_name=drug,
                contraindications_for_drug=drug_contraindications.get(drug, []),
                patient_state=_patient_ctx,
                lab_context=lab_context,
                active_conditions=active_conditions,
                current_medications=[{"drug": m} for m in current_medications],
                allergies=patient_allergies,
            )
            for drug in flagged
        ])
        for drug, result in zip(flagged, alt_results):
            proactive_alternatives[drug] = result.model_dump() if result else None
            if result:
                logger.info(f"  Alternatives for {drug}: {[a.drug for a in result.alternatives]}")

    # ── Phase 5c: Merge enrichment ────────────────────────────────────────────
    enriched_rxnav = list(rxnav_interactions)
    if interaction_enrichment and interaction_enrichment.enriched_interactions:
        for i, e in enumerate(interaction_enrichment.enriched_interactions):
            if i < len(enriched_rxnav):
                enriched_rxnav[i] = {
                    **enriched_rxnav[i],
                    "mechanism":             e.mechanism,
                    "clinical_significance": e.clinical_significance,
                    "monitoring_parameters": e.monitoring_parameters,
                    "management_strategy":   e.management_strategy,
                    "time_to_onset":         e.time_to_onset,
                }

    interaction_risk_narrative = (
        interaction_enrichment.overall_risk_narrative if interaction_enrichment else None
    )

    # ── Phase 5d: FHIR assembly — drug-specific notes ─────────────────────────
    risk_level = (patient_risk_profile or {}).get("overall_risk_level", "LOW")
    lab_note = lab_context.overall_lab_summary if lab_context.critical_flags else ""

    def _get_specific_reason(drug: str) -> str:
        drug_contras = drug_contraindications.get(drug, [])
        if not drug_contras:
            return "Safety concern identified."
        top = max(drug_contras,
                  key=lambda c: {"CRITICAL": 3, "HIGH": 2, "MODERATE": 1, "LOW": 0}.get(c.get("severity", "LOW"), 0))
        reason = top.get("reason", "")
        drug_base = drug.lower().split()[0]
        for i in enriched_rxnav:
            if drug_base in i.get("drug_a", "").lower() or drug_base in i.get("drug_b", "").lower():
                other = i.get("drug_b") if drug_base in i.get("drug_a", "").lower() else i.get("drug_a")
                reason += f" Additionally: {i.get('severity')} interaction with {other}."
                break
        return reason

    fhir_medication_requests = []
    for drug in approved:
        interaction_note = next(
            (f"{i.get('severity')} interaction with "
             f"{i.get('drug_b') if drug.lower().split()[0] in i.get('drug_a','').lower() else i.get('drug_a')} — monitor."
             for i in enriched_rxnav
             if drug.lower().split()[0] in i.get("drug_a", "").lower()
             or drug.lower().split()[0] in i.get("drug_b", "").lower()),
            ""
        )
        fhir_medication_requests.append(
            build_fhir_medication_request(drug, patient_id, safety_cleared=True,
                                          safety_note=interaction_note, risk_level=risk_level,
                                          lab_context_note=lab_note)
        )
    for drug in flagged:
        fhir_medication_requests.append(
            build_fhir_medication_request(drug, patient_id, safety_cleared=False,
                                          risk_level=risk_level,
                                          specific_reason=_get_specific_reason(drug),
                                          lab_context_note=lab_note)
        )

    # ── Phase 6: Persist ──────────────────────────────────────────────────────
    request_id = str(uuid.uuid4())[:8]
    if persist:
        await save_drug_safety(DrugSafetyRecord(
            request_id=request_id, patient_id=patient_id,
            proposed_medications=proposed_medications, current_medications=current_medications,
            safety_status=safety_status, contraindications=contraindications,
            approved_medications=approved, flagged_medications=flagged,
            critical_interactions=enriched_rxnav, fda_warnings=fda_warnings,
            patient_risk_profile=patient_risk_profile,
            interaction_risk_narrative=interaction_risk_narrative,
            fhir_medication_requests=fhir_medication_requests,
            llm_enriched=enrich_with_llm and risk_profile is not None,
            cache_hit=False, source=source,
        ))

    result = {
        "safety_status": safety_status,
        "request_id":    request_id,
        "lab_assessment": {
            "critical_flags":              [f.model_dump() for f in lab_context.critical_flags],
            "sepsis_suspicion":            lab_context.sepsis_suspicion,
            "renal_impairment_suspected":  lab_context.renal_impairment_suspected,
            "hepatic_impairment_suspected":lab_context.hepatic_impairment_suspected,
            "coagulopathy_suspected":      lab_context.coagulopathy_suspected,
            "overall_lab_summary":         lab_context.overall_lab_summary,
            "drug_selection_constraints":  lab_context.drug_selection_constraints,
        },
        "contraindications":        contraindications,
        "critical_interactions":    enriched_rxnav,
        "fda_warnings":             fda_warnings,
        "rxcui_map":                rxcui_map,
        "approved_medications":     approved,
        "flagged_medications":      flagged,
        "fhir_medication_requests": fhir_medication_requests,
        "proactive_alternatives":   proactive_alternatives,
        "patient_risk_profile":     patient_risk_profile,
        "interaction_risk_narrative": interaction_risk_narrative,
        "summary": {
            "proposed_count":         len(proposed_medications),
            "approved_count":         len(approved),
            "flagged_count":          len(flagged),
            "interaction_count":      len(rxnav_interactions),
            "contraindication_count": len(contraindications),
            "black_box_warnings":     sum(1 for w in fda_warnings.values() if any("[BLACK BOX]" in x for x in w)),
            "severity_upgrades":      len([i for i in rxnav_interactions if i.get("severity_upgraded")]),
            "critical_lab_flags":     len(lab_context.critical_flags),
            "sepsis_suspicion":       lab_context.sepsis_suspicion,
            "alternatives_generated": len([v for v in proactive_alternatives.values() if v]),
            "llm_enriched":           enrich_with_llm and risk_profile is not None,
            "llm_vetoed":             llm_vetoed,
        },
    }

    base_result = {k: v for k, v in result.items()
                   if k not in ("patient_risk_profile", "interaction_risk_narrative", "proactive_alternatives")}
    await cache_set(cache_key, base_result, ttl=3600)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FastMCP — MCP Tools
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="drug-safety-mcp",
    instructions="Drug safety checker v3.0 — lab context, severity overrides, proactive alternatives.",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def check_drug_interactions(
    medications: Annotated[list[str], Field(description="All medications to check. Include current + proposed.")],
    patient_id: Annotated[Optional[str], Field(description="Optional FHIR patient ID.")] = None,
) -> dict:
    """Check drug-drug interactions with FDA labels + severity overrides + LLM enrichment."""
    if len(medications) < 2:
        return {"interactions": [], "message": "Need at least 2 medications."}
    interactions = await check_drug_interactions_by_name(medications)
    interactions = apply_severity_overrides(interactions)
    minimal_state = {"demographics": {"age":"unknown","gender":"unknown"}, "active_conditions":[], "medications":[{"drug":m} for m in medications], "allergies":[]}
    enrichment = await enrich_interactions_with_llm(interactions, minimal_state)
    enriched = list(interactions)
    if enrichment:
        for i, e in enumerate(enrichment.enriched_interactions):
            if i < len(enriched):
                enriched[i] = {**enriched[i], "mechanism": e.mechanism, "clinical_significance": e.clinical_significance, "monitoring_parameters": e.monitoring_parameters, "management_strategy": e.management_strategy, "time_to_onset": e.time_to_onset}
    return {"medications_checked": medications, "interactions": enriched, "interaction_count": len(interactions), "high_severity_count": sum(1 for i in interactions if i.get("severity","").upper() in ("HIGH","CRITICAL")), "overall_risk_narrative": enrichment.overall_risk_narrative if enrichment else None}


@mcp.tool()
async def get_contraindications(
    drug_name: Annotated[str, Field(description="Drug name to check.")],
    conditions: Annotated[Optional[list[str]], Field(description="Patient ICD-10 codes.")] = None,
    allergies: Annotated[Optional[list[str]], Field(description="Known allergens.")] = None,
) -> dict:
    """Check allergy cross-reactivity, condition contraindications, and FDA black box warnings."""
    allergy_dicts = [{"substance": a, "reaction": "unknown", "severity": "unknown"} for a in (allergies or [])]
    condition_dicts = [{"code": c, "display": c} for c in (conditions or [])]
    allergy_contras, condition_contras, fda_w = await asyncio.gather(
        asyncio.to_thread(check_allergy_cross_reactivity, drug_name, allergy_dicts),
        asyncio.to_thread(check_condition_contraindications, drug_name, condition_dicts),
        get_fda_warnings_batch([drug_name]),
    )
    all_contras = allergy_contras + condition_contras
    has_critical = any(c.get("severity") in ("CRITICAL", "HIGH") for c in all_contras)
    drug_fda = fda_w.get(drug_name, [])
    return {"drug": drug_name, "overall_verdict": "UNSAFE" if has_critical else ("CAUTION" if all_contras or drug_fda else "SAFE"), "contraindications": all_contras, "contraindication_count": len(all_contras), "allergy_contraindications": allergy_contras, "condition_contraindications": condition_contras, "fda_warnings": drug_fda, "has_black_box_warning": any("[BLACK BOX]" in w for w in drug_fda)}


@mcp.tool()
async def suggest_alternatives(
    drug_name: Annotated[str, Field(description="Drug that cannot be prescribed.")],
    reason_for_avoidance: Annotated[str, Field(description="Why this drug cannot be used.")],
    indication: Annotated[str, Field(description="What the drug was meant to treat.")],
    patient_conditions: Annotated[Optional[list[str]], Field(description="Patient ICD-10 codes.")] = None,
    current_medications: Annotated[Optional[list[str]], Field(description="Current medications.")] = None,
    patient_allergies: Annotated[Optional[list[str]], Field(description="Known allergens.")] = None,
) -> dict:
    """Suggest safe alternatives with LLM clinical reasoning."""
    result = await get_alternative_suggestions_async(
        drug_name=drug_name, reason_for_avoidance=reason_for_avoidance, indication=indication,
        patient_conditions=[{"code": c, "display": c} for c in (patient_conditions or [])],
        current_medications=[{"drug": m} for m in (current_medications or [])],
        allergies=[{"substance": a, "reaction": "unknown", "severity": "unknown"} for a in (patient_allergies or [])],
    )
    if not result:
        return {"drug_to_replace": drug_name, "alternatives": [], "error": "LLM unavailable."}
    return {"drug_to_replace": drug_name, "reason_for_avoidance": reason_for_avoidance, "indication": indication, "alternatives": [a.model_dump() for a in result.alternatives], "clinical_note": result.clinical_note, "urgency": result.urgency}


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI REST
# ══════════════════════════════════════════════════════════════════════════════

class SafetyCheckRequest(BaseModel):
    proposed_medications: list[str] = Field(example=["Amoxicillin 500mg", "Ibuprofen 400mg"])
    current_medications:  list[str] = Field(default_factory=list)
    patient_allergies:    list[dict] = Field(default_factory=list)
    active_conditions:    list[dict] = Field(default_factory=list)
    patient_id:           str = Field(default="unknown")
    patient_state:        Optional[dict] = Field(default=None)
    enrich_with_llm:      bool = Field(default=True)


rest_app = FastAPI(title="MediTwin Drug Safety Agent", version="3.0.0")
rest_app.include_router(stream_router)
rest_app.include_router(history_router, prefix="/history", tags=["history"])

from fastapi.middleware.cors import CORSMiddleware
rest_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@rest_app.post("/check-safety")
async def check_safety(request: SafetyCheckRequest) -> JSONResponse:
    result = await _run_full_safety_check(
        proposed_medications=request.proposed_medications,
        current_medications=request.current_medications,
        patient_allergies=request.patient_allergies,
        active_conditions=request.active_conditions,
        patient_id=request.patient_id,
        patient_state=request.patient_state,
        enrich_with_llm=request.enrich_with_llm,
        persist=True, source="check-safety",
    )
    return JSONResponse(content=result)


@rest_app.get("/health")
async def health() -> JSONResponse:
    r = await get_redis()
    return JSONResponse(content={
        "status": "healthy", "agent": "drug-safety", "version": "3.0.0",
        "redis": "connected" if r else "disconnected",
        "db": "connected" if db.is_available() else "disconnected",
        "pipeline_phases": [
            "0. Critical lab assessment (WBC/Cr/INR/K+/ALT/Hgb/eGFR — deterministic)",
            "1. Allergy cross-reactivity + condition contraindications + lab-driven checks",
            "2. FDA label interactions + INTERACTION_SEVERITY_OVERRIDE table (20+ pairs)",
            "3. Safety status",
            "4. LLM enrichment with lab context",
            "5a. LLM veto filter",
            "5b. Proactive alternatives for every flagged drug (concurrent, with sepsis/renal context)",
            "5c. Drug-specific FHIR notes",
            "6. DB persistence",
        ],
    })


@asynccontextmanager
async def _lifespan(app: Starlette):
    await db_init()
    yield
    await db_close()


mcp_asgi = mcp.streamable_http_app()
combined_app = Starlette(
    routes=[Mount("/mcp", app=mcp_asgi), Mount("/", app=rest_app)],
    lifespan=_lifespan,
)

# Alias for uvicorn (uvicorn main:app)
app = combined_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="127.0.0.1", port=8004)