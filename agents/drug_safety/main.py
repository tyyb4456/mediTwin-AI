"""
Agent 4: Drug Safety MCP Server — v2.0
Port: 8004

DUAL PURPOSE:
  1. MCP Superpower — published to Prompt Opinion Marketplace.
     Three tools: check_drug_interactions, get_contraindications, suggest_alternatives

  2. A2A component inside MediTwin — Orchestrator calls /check-safety (REST)

CHANGES IN v2.0:
  - All LLM calls use llm.with_structured_output(PydanticModel) — no more JsonOutputParser
  - FDA black box warnings fetched and integrated into safety pipeline
  - LLM enriches interactions with mechanism, monitoring params, management strategy
  - LLM generates patient-specific risk profile synthesizing ALL safety signals
  - suggest_alternatives MCP tool is now properly async end-to-end
  - Rich structured response includes patient_risk_profile and enriched_interactions
  - Renal/hepatic impairment detection from conditions informs LLM context
  - FHIR MedicationRequest output includes priority and AI extension

Architecture:
  FastMCP (stateless HTTP) mounted at /mcp/
  FastAPI REST endpoints at / (root)
  Single combined Starlette ASGI app

Safety pipeline order (all runs in parallel where possible):
  1. Local checks: allergy cross-reactivity + condition contraindications (deterministic, fast)
  2. External API: RxNav interaction check + FDA warnings fetch (concurrent)
  3. Determine preliminary safety_status from deterministic findings
  4. LLM enrichment: interaction clinical context + patient risk profile (concurrent, additive)
  5. Assemble final response
"""
import os
import sys
import json
import asyncio
from typing import Optional, Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    build_fhir_medication_request,
)


# ── Redis cache setup ─────────────────────────────────────────────────────────

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
            print(f"  ⚠️  Redis unavailable ({e}) — caching disabled")
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
# Core safety pipeline (shared by MCP tools AND REST endpoint)
# ══════════════════════════════════════════════════════════════════════════════

async def _run_full_safety_check(
    proposed_medications: list[str],
    current_medications: list[str],
    patient_allergies: list[dict],   # [{substance, reaction, severity}]
    active_conditions: list[dict],   # [{code, display}]
    patient_id: str = "unknown",
    patient_state: Optional[dict] = None,   # Full PatientState for LLM enrichment
    enrich_with_llm: bool = True,
) -> dict:
    """
    Full drug safety pipeline.

    Phase 1 — Deterministic (fast, always runs):
      1a. Allergy cross-reactivity check (local table)
      1b. Condition contraindication check (local JSON)

    Phase 2 — External APIs (concurrent):
      2a. NLM RxNav drug-drug interaction check
      2b. FDA OpenFDA black box / label warnings

    Phase 3 — Safety status (from deterministic findings)

    Phase 4 — LLM enrichment (concurrent, additive — never overrides rules):
      4a. Clinical interaction enrichment (mechanism, monitoring, management)
      4b. Patient-specific risk profile (synthesizes all signals)

    Phase 5 — Assemble response
    """
    all_drugs = proposed_medications + current_medications
    # Cache key: drug combo + allergies (LLM enrichment NOT cached — patient-specific)
    cache_key = (
        f"drug_safety_v2:"
        f"{':'.join(sorted(all_drugs))}:"
        f"{':'.join(sorted(a.get('substance','') for a in patient_allergies))}"
    )

    # Only cache the non-LLM parts (deterministic + external API results)
    base_cached = await cache_get(cache_key)

    if base_cached and not enrich_with_llm:
        print("  ✓ Full cache hit")
        return base_cached
# rxnav_interactions 
    # ── Phase 1: Deterministic local checks ──────────────────────────────────
    print("  Phase 1: Running deterministic safety checks...")
    contraindications = []
    approved = []
    flagged = []

    for drug in proposed_medications:
        drug_contras = []
        drug_contras.extend(check_allergy_cross_reactivity(drug, patient_allergies))
        drug_contras.extend(check_condition_contraindications(drug, active_conditions))

        if drug_contras:
            contraindications.extend(drug_contras)
            flagged.append(drug)
        else:
            approved.append(drug)

    # ── Phase 2: Concurrent external API calls ────────────────────────────────
    print("  Phase 2: Fetching external safety data (RxNav + FDA)...")

    rxcui_map: dict = {}
    rxnav_interactions: list = []
    fda_warnings: dict = {}

    if len(all_drugs) >= 2:
        rxcui_map, fda_warnings = await asyncio.gather(
            get_rxcuis_batch(all_drugs),
            get_fda_warnings_batch(proposed_medications),  # Only check proposed drugs
        )
        rxcuis = [v for v in rxcui_map.values() if v]
        if len(rxcuis) >= 2:
            print(f"  Resolved {len(rxcuis)}/{len(all_drugs)} drugs to RxCUI — checking interactions...")
            rxnav_interactions = await check_drug_interactions_by_name(all_drugs)
        else:
            print(f"  ⚠️  Only {len(rxcuis)} RxCUI(s) resolved — skipping RxNav interaction check")
    else:
        fda_warnings = await get_fda_warnings_batch(proposed_medications)

    # ── Phase 3: Determine safety status ─────────────────────────────────────
    has_critical_contra = any(
        c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications
    )
    has_severe_interaction = any(
        i.get("severity", "").upper() in ("HIGH", "CRITICAL")
        for i in rxnav_interactions
    )
    has_black_box = any(
        any("[BLACK BOX]" in w for w in warnings)
        for warnings in fda_warnings.values()
        if warnings
    )

    if has_critical_contra or has_severe_interaction:
        safety_status = "UNSAFE"
    elif contraindications or rxnav_interactions or has_black_box:
        safety_status = "CAUTION"
    else:
        safety_status = "SAFE"

    # ── Phase 4: LLM enrichment (concurrent, non-blocking) ───────────────────
    enriched_interactions = None
    patient_risk_profile = None
    interaction_enrichment = None   # ← ADD THIS LINE
    risk_profile = None      # ← ADD THIS LINE

    # Build a minimal patient_state if not provided (for LLM context)
    _patient_ctx = patient_state or {
        "demographics": {"age": "unknown", "gender": "unknown"},
        "active_conditions": active_conditions,
        "medications": [{"drug": m} for m in current_medications],
        "allergies": patient_allergies,
    }

    if enrich_with_llm:
        print("  Phase 4: LLM enrichment (interactions + risk profile)...")
        llm_tasks = [
            enrich_interactions_with_llm(rxnav_interactions, _patient_ctx),
            generate_patient_risk_profile(
                patient_state=_patient_ctx,
                proposed_medications=proposed_medications,
                contraindications=contraindications,
                interactions=rxnav_interactions,
                fda_warnings=fda_warnings,
                safety_status=safety_status,
            ),
        ]
        interaction_enrichment, risk_profile = await asyncio.gather(*llm_tasks)

        if interaction_enrichment:
            enriched_interactions = interaction_enrichment.model_dump()
        if risk_profile:
            patient_risk_profile = risk_profile.model_dump()

    # ── Phase 5: Assemble FHIR MedicationRequests for cleared drugs ───────────
    risk_level = (
        patient_risk_profile.get("overall_risk_level", "LOW")
        if patient_risk_profile else "LOW"
    )

    fhir_medication_requests = []
    for drug in approved:
        interaction_note = ""
        if any(i.get("severity", "").upper() == "MODERATE" for i in rxnav_interactions):
            interaction_note = "Moderate drug interaction detected — monitor per clinician guidance."
        fhir_medication_requests.append(
            build_fhir_medication_request(
                drug, patient_id,
                safety_cleared=True,
                safety_note=interaction_note,
                risk_level=risk_level,
            )
        )

    # ── Merge LLM enrichment into interaction list ───────────────────────────
    enriched_rxnav = list(rxnav_interactions)
    if interaction_enrichment and interaction_enrichment.enriched_interactions:
        for i, enrichment in enumerate(interaction_enrichment.enriched_interactions):
            if i < len(enriched_rxnav):
                enriched_rxnav[i] = {
                    **enriched_rxnav[i],
                    "mechanism":             enrichment.mechanism,
                    "clinical_significance": enrichment.clinical_significance,
                    "monitoring_parameters": enrichment.monitoring_parameters,
                    "management_strategy":   enrichment.management_strategy,
                    "time_to_onset":         enrichment.time_to_onset,
                }

    result = {
        "safety_status": safety_status,

        # Core safety findings (deterministic)
        "contraindications":       contraindications,
        "critical_interactions":   enriched_rxnav,

        # External data
        "fda_warnings":            fda_warnings,
        "rxcui_map":               rxcui_map,

        # Medication verdicts
        "approved_medications":    approved,
        "flagged_medications":     flagged,
        "fhir_medication_requests": fhir_medication_requests,

        # LLM-generated patient-specific assessment
        "patient_risk_profile": patient_risk_profile,
        "interaction_risk_narrative": (
            interaction_enrichment.overall_risk_narrative
            if interaction_enrichment else None
        ),

        # Summary counts
        "summary": {
            "proposed_count":       len(proposed_medications),
            "approved_count":       len(approved),
            "flagged_count":        len(flagged),
            "interaction_count":    len(rxnav_interactions),
            "contraindication_count": len(contraindications),
            "black_box_warnings":   sum(
                1 for w in fda_warnings.values()
                if any("[BLACK BOX]" in x for x in w)
            ),
            "llm_enriched":         enrich_with_llm and patient_risk_profile is not None,
        },
    }

    # Cache the base result (without LLM — patient-specific profiles shouldn't be cached)
    base_result = {k: v for k, v in result.items()
                   if k not in ("patient_risk_profile", "interaction_risk_narrative")}
    await cache_set(cache_key, base_result, ttl=3600)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FastMCP — 3 MCP Tools
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="drug-safety-mcp",
    instructions=(
        "Drug safety checker for clinical decision support. "
        "Checks drug-drug interactions (NLM RxNav), allergy cross-reactivity, "
        "condition contraindications, and FDA black box warnings. "
        "Returns safety verdicts, enriched clinical context, and safe alternatives. "
        "SHARP context: pass patient_id to enable automatic patient data fetching."
    ),
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def check_drug_interactions(
    medications: Annotated[
        list[str],
        Field(
            description=(
                "List of drug names to check for interactions. "
                "Include both proposed and current medications. "
                "Example: ['Amoxicillin 500mg', 'Warfarin 5mg', 'Metformin 850mg']"
            )
        )
    ],
    patient_id: Annotated[
        Optional[str],
        Field(description="Optional FHIR patient ID. Enables SHARP context passthrough.")
    ] = None,
) -> dict:
    """
    Check for dangerous drug-drug interactions between a list of medications.
    Uses FDA drug label warnings to detect cross-drug mentions.
    Returns interactions with severity levels and LLM-enriched clinical context
    (mechanism, monitoring parameters, management strategy).
    """
    if len(medications) < 2:
        return {"interactions": [], "message": "Need at least 2 medications to check interactions."}

    interactions = await check_drug_interactions_by_name(medications)

    # LLM enrichment — build minimal patient context
    minimal_state = {
        "demographics": {"age": "unknown", "gender": "unknown"},
        "active_conditions": [],
        "medications": [{"drug": m} for m in medications],
        "allergies": [],
    }
    enrichment = await enrich_interactions_with_llm(interactions, minimal_state)

    enriched = list(interactions)
    if enrichment:
        for i, e in enumerate(enrichment.enriched_interactions):
            if i < len(enriched):
                enriched[i] = {
                    **enriched[i],
                    "mechanism": e.mechanism,
                    "clinical_significance": e.clinical_significance,
                    "monitoring_parameters": e.monitoring_parameters,
                    "management_strategy": e.management_strategy,
                    "time_to_onset": e.time_to_onset,
                }

    return {
        "medications_checked":    medications,
        "interactions":           enriched,
        "interaction_count":      len(interactions),
        "high_severity_count":    sum(
            1 for i in interactions if i.get("severity", "").upper() in ("HIGH", "CRITICAL")
        ),
        "overall_risk_narrative": enrichment.overall_risk_narrative if enrichment else None,
        "source":                 "FDA Drug Label Warnings + LLM Clinical Enrichment",
    }


@mcp.tool()
async def get_contraindications(
    drug_name: Annotated[
        str,
        Field(description="Drug name to check. Include dose if known, e.g. 'Levofloxacin 750mg'.")
    ],
    conditions: Annotated[
        Optional[list[str]],
        Field(description="Patient's active ICD-10 condition codes. Example: ['N18.4', 'I50.9']")
    ] = None,
    allergies: Annotated[
        Optional[list[str]],
        Field(description="Patient's known allergens. Example: ['Penicillin', 'Sulfa']")
    ] = None,
) -> dict:
    """
    Check if a medication has contraindications for a patient's conditions or allergies.
    Checks allergy cross-reactivity and condition contraindications.
    Also fetches FDA black box warnings for the drug.
    """
    allergy_dicts = [
        {"substance": a, "reaction": "unknown", "severity": "unknown"}
        for a in (allergies or [])
    ]
    condition_dicts = [{"code": c, "display": c} for c in (conditions or [])]

    # Run local checks + FDA warning fetch concurrently
    allergy_contras, condition_contras, fda_w = await asyncio.gather(
        asyncio.to_thread(check_allergy_cross_reactivity, drug_name, allergy_dicts),
        asyncio.to_thread(check_condition_contraindications, drug_name, condition_dicts),
        get_fda_warnings_batch([drug_name]),
    )

    all_contras = allergy_contras + condition_contras
    has_critical = any(c.get("severity") in ("CRITICAL", "HIGH") for c in all_contras)
    drug_fda_warnings = fda_w.get(drug_name, [])

    overall_verdict = (
        "UNSAFE" if has_critical else
        ("CAUTION" if all_contras or drug_fda_warnings else "SAFE")
    )

    return {
        "drug":                        drug_name,
        "overall_verdict":             overall_verdict,
        "contraindications":           all_contras,
        "contraindication_count":      len(all_contras),
        "allergy_contraindications":   allergy_contras,
        "condition_contraindications": condition_contras,
        "fda_warnings":                drug_fda_warnings,
        "has_black_box_warning":       any("[BLACK BOX]" in w for w in drug_fda_warnings),
    }


@mcp.tool()
async def suggest_alternatives(
    drug_name: Annotated[
        str,
        Field(description="The drug that cannot be prescribed. Example: 'Amoxicillin 500mg'")
    ],
    reason_for_avoidance: Annotated[
        str,
        Field(description="Why this drug cannot be used. Example: 'Penicillin allergy — anaphylaxis'")
    ],
    indication: Annotated[
        str,
        Field(description="What condition the drug was meant to treat. Example: 'Community-acquired pneumonia'")
    ],
    patient_conditions: Annotated[
        Optional[list[str]],
        Field(description="Patient's ICD-10 condition codes. Example: ['J18.9', 'N18.3']")
    ] = None,
    current_medications: Annotated[
        Optional[list[str]],
        Field(description="Patient's current medications. Example: ['Warfarin 5mg', 'Metformin 850mg']")
    ] = None,
    patient_allergies: Annotated[
        Optional[list[str]],
        Field(description="Known allergens to exclude. Example: ['Penicillin', 'Sulfa']")
    ] = None,
) -> dict:
    """
    Suggest safe medication alternatives when a drug has contraindications or interactions.
    Uses LLM with structured output for clinical reasoning.
    Returns ranked alternatives with rationale, drug class, interaction checks needed, and urgency.
    """
    condition_dicts = [{"code": c, "display": c} for c in (patient_conditions or [])]
    med_dicts = [{"drug": m} for m in (current_medications or [])]
    allergy_dicts = [
        {"substance": a, "reaction": "unknown", "severity": "unknown"}
        for a in (patient_allergies or [])
    ]

    result = await get_alternative_suggestions_async(
        drug_name=drug_name,
        reason_for_avoidance=reason_for_avoidance,
        indication=indication,
        patient_conditions=condition_dicts,
        current_medications=med_dicts,
        allergies=allergy_dicts,
    )

    if not result:
        return {
            "drug_to_replace": drug_name,
            "alternatives": [],
            "error": "LLM alternative suggestion unavailable. GOOGLE_API_KEY not set or API error.",
        }

    return {
        "drug_to_replace":     drug_name,
        "reason_for_avoidance": reason_for_avoidance,
        "indication":          indication,
        "alternatives":        [a.model_dump() for a in result.alternatives],
        "clinical_note":       result.clinical_note,
        "urgency":             result.urgency,
        "source":              "LLM Clinical Pharmacist (Gemini 2.5 Flash)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI REST — for MediTwin Orchestrator internal calls
# ══════════════════════════════════════════════════════════════════════════════

class SafetyCheckRequest(BaseModel):
    proposed_medications: list[str] = Field(
        description="Drugs being considered for prescription",
        example=["Amoxicillin 500mg", "Ibuprofen 400mg"]
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Drugs patient is already taking",
        example=["Warfarin 5mg", "Metformin 850mg"]
    )
    patient_allergies: list[dict] = Field(
        default_factory=list,
        description="List of {substance, reaction, severity} dicts"
    )
    active_conditions: list[dict] = Field(
        default_factory=list,
        description="List of {code, display} ICD-10 condition dicts"
    )
    patient_id: str = Field(default="unknown")
    patient_state: Optional[dict] = Field(
        default=None,
        description="Full PatientState object for richer LLM context (optional)"
    )
    enrich_with_llm: bool = Field(
        default=True,
        description="Set False to skip LLM enrichment (faster, for testing)"
    )


rest_app = FastAPI(
    title="MediTwin Drug Safety Agent",
    description="Drug Safety MCP Server (v2.0) + REST API. LLM-enriched safety pipeline.",
    version="2.0.0",
)


@rest_app.post("/check-safety")
async def check_safety(request: SafetyCheckRequest) -> JSONResponse:
    """
    Full drug safety check — called by MediTwin Orchestrator.

    Pipeline:
      1. Allergy cross-reactivity (deterministic)
      2. Condition contraindications (deterministic)
      3. RxNav drug-drug interactions (external API)
      4. FDA black box warnings (external API, concurrent with RxNav)
      5. LLM interaction enrichment (structured Pydantic output)
      6. LLM patient risk profile (structured Pydantic output)
      7. FHIR MedicationRequest generation for cleared drugs
    """
    result = await _run_full_safety_check(
        proposed_medications=request.proposed_medications,
        current_medications=request.current_medications,
        patient_allergies=request.patient_allergies,
        active_conditions=request.active_conditions,
        patient_id=request.patient_id,
        patient_state=request.patient_state,
        enrich_with_llm=request.enrich_with_llm,
    )
    return JSONResponse(content=result)


@rest_app.get("/health")
async def health() -> JSONResponse:
    r = await get_redis()
    return JSONResponse(content={
        "status":        "healthy",
        "agent":         "drug-safety",
        "version":       "2.0.0",
        "redis":         "connected" if r else "disconnected",
        "mcp_tools":     ["check_drug_interactions", "get_contraindications", "suggest_alternatives"],
        "mcp_endpoint":  "/mcp/",
        "rest_endpoint": "/check-safety",
        "llm_model":     "gemini-2.5-flash (structured output via .with_structured_output())",
        "external_apis": ["NLM RxNav", "NLM RxNorm", "FDA OpenFDA"],
        "local_databases": ["cross_reactivity_table", "contraindications.json"],
        "pipeline_phases": [
            "1. Allergy cross-reactivity (deterministic)",
            "2. Condition contraindications (deterministic)",
            "3. RxNav interactions (async API)",
            "4. FDA warnings (async API, concurrent)",
            "5. LLM interaction enrichment (structured output)",
            "6. LLM patient risk profile (structured output)",
            "7. FHIR MedicationRequest assembly",
        ],
    })


# ── Mount MCP + REST into unified ASGI app ───────────────────────────────────

mcp_asgi = mcp.streamable_http_app()

combined_app = Starlette(
    routes=[
        Mount("/mcp", app=mcp_asgi),
        Mount("/",    app=rest_app),
    ]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="127.0.0.1", port=8004)