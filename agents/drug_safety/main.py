"""
Agent 4: Drug Safety MCP Server
Port: 8004

DUAL PURPOSE:
  1. MCP Superpower — published to Prompt Opinion Marketplace.
     Any agent on the platform can call our 3 tools:
       - check_drug_interactions
       - get_contraindications
       - suggest_alternatives

  2. A2A component inside MediTwin — the Orchestrator calls /check-safety
     (REST endpoint) with the full patient context. The REST handler calls
     the same core logic as the MCP tools, so there is ONE code path.

Architecture decision:
  - FastMCP for the MCP protocol layer (HTTP transport, recommended for 2025)
  - FastAPI mounted alongside for the /check-safety REST endpoint + /health
  - Redis for caching drug interaction results (TTL: 1 hour)
  - All external API calls go through fda_client.py (async httpx)
  - All local lookups go through safety_core.py (pure Python, no LLM)
  - LLM only used for suggest_alternatives tool

Endpoints:
  MCP:  POST /mcp/          (FastMCP streamable-HTTP)
  REST: POST /check-safety  (Orchestrator calls this)
  REST: GET  /health
"""
import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from starlette.routing import Mount

# Add parent directories to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fda_client import get_rxcuis_batch, check_rxnav_interactions
from safety_core import (
    check_allergy_cross_reactivity,
    check_condition_contraindications,
    get_alternative_suggestions,
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


# ── Core safety check function (shared by MCP tools and REST endpoint) ─────────

async def _run_full_safety_check(
    proposed_medications: list[str],
    current_medications: list[str],
    patient_allergies: list[dict],   # [{substance, reaction, severity}]
    active_conditions: list[dict],   # [{code, display}]
    patient_id: str = "unknown",
) -> dict:
    """
    Full drug safety pipeline. Called by both MCP tools and REST endpoint.

    Steps:
      1. Check allergy cross-reactivity for each proposed drug (local, fast)
      2. Check condition contraindications for each proposed drug (local, fast)
      3. Check drug-drug interactions via NLM RxNav (async API call)
      4. Determine overall safety status
      5. Build FHIR MedicationRequest resources for cleared drugs
    """
    all_drugs = proposed_medications + current_medications
    cache_key = f"drug_safety:{':'.join(sorted(all_drugs))}:{':'.join(sorted(a.get('substance','') for a in patient_allergies))}"

    # Cache hit?
    cached = await cache_get(cache_key)
    if cached:
        print(f"  ✓ Cache hit for drug safety check")
        return cached

    critical_interactions = []
    contraindications = []
    approved = []
    flagged = []

    # ── Step 1 + 2: Local checks for each proposed drug ─────────────────────
    for drug in proposed_medications:
        drug_contraindications = []

        # Allergy cross-reactivity
        allergy_contras = check_allergy_cross_reactivity(drug, patient_allergies)
        drug_contraindications.extend(allergy_contras)

        # Condition contraindications
        condition_contras = check_condition_contraindications(drug, active_conditions)
        drug_contraindications.extend(condition_contras)

        if drug_contraindications:
            contraindications.extend(drug_contraindications)
            flagged.append(drug)
        else:
            approved.append(drug)

    # ── Step 3: Drug-drug interaction check via RxNav ─────────────────────────
    if len(all_drugs) >= 2:
        print(f"  Resolving RxCUIs for {all_drugs}...")
        rxcui_map = await get_rxcuis_batch(all_drugs)
        rxcuis = [v for v in rxcui_map.values() if v]

        if len(rxcuis) >= 2:
            print(f"  Checking RxNav interactions for {len(rxcuis)} RxCUIs...")
            rxnav_interactions = await check_rxnav_interactions(rxcuis)
            critical_interactions.extend(rxnav_interactions)
        else:
            print(f"  ⚠️  Only {len(rxcuis)} RxCUI(s) resolved — skipping interaction check")
    else:
        rxnav_interactions = []

    # ── Step 4: Determine overall safety status ───────────────────────────────
    has_critical_contra = any(
        c.get("severity") in ("CRITICAL", "HIGH") for c in contraindications
    )
    has_severe_interaction = any(
        i.get("severity", "").upper() in ("HIGH", "CRITICAL")
        for i in critical_interactions
    )

    safety_status = "UNSAFE" if (has_critical_contra or has_severe_interaction) else "SAFE"
    if contraindications and not has_critical_contra:
        safety_status = "CAUTION"  # Low-severity issues — proceed with monitoring

    # ── Step 5: FHIR MedicationRequests for approved drugs ───────────────────
    fhir_medication_requests = []
    for drug in approved:
        note = "No contraindications or allergies detected."
        if any(i.get("severity", "").upper() == "MODERATE" for i in critical_interactions):
            note = "Moderate drug interaction detected — monitor closely."
        fhir_medication_requests.append(
            build_fhir_medication_request(drug, patient_id, safety_cleared=True, safety_note=note)
        )

    result = {
        "safety_status": safety_status,
        "critical_interactions": critical_interactions,
        "contraindications": contraindications,
        "approved_medications": approved,
        "flagged_medications": flagged,
        "fhir_medication_requests": fhir_medication_requests,
        "summary": {
            "proposed_count": len(proposed_medications),
            "approved_count": len(approved),
            "flagged_count": len(flagged),
            "interaction_count": len(critical_interactions),
            "contraindication_count": len(contraindications),
        },
    }

    # Cache for 1 hour
    await cache_set(cache_key, result, ttl=3600)
    return result


# ── FastMCP: MCP Server with 3 tools ─────────────────────────────────────────

mcp = FastMCP(
    name="drug-safety-mcp",
    instructions=(
        "Drug safety checker for clinical decision support. "
        "Checks drug-drug interactions using NLM RxNav, allergy cross-reactivity, "
        "and condition contraindications. Returns safety verdicts and safe alternatives. "
        "SHARP context: pass patient_id to enable automatic medication list fetching."
    ),
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def check_drug_interactions(
    medications: Annotated[
        list[str],
        Field(description="List of drug names to check for interactions. Include both proposed and current medications. Example: ['Amoxicillin 500mg', 'Warfarin 5mg', 'Metformin 850mg']")
    ],
    patient_id: Annotated[
        Optional[str],
        Field(description="Optional FHIR patient ID. When provided with SHARP context, enables automatic patient medication list retrieval.")
    ] = None,
) -> dict:
    """
    Check for dangerous drug-drug interactions between a list of medications.
    Uses NLM RxNav interaction database (free, no auth required).
    Returns interactions with severity levels (HIGH/MODERATE/LOW) and clinical recommendations.
    """
    if len(medications) < 2:
        return {"interactions": [], "message": "Need at least 2 medications to check interactions."}

    rxcui_map = await get_rxcuis_batch(medications)
    rxcuis = [v for v in rxcui_map.values() if v]

    if len(rxcuis) < 2:
        return {
            "interactions": [],
            "message": f"Could only resolve {len(rxcuis)} drug(s) to RxCUI codes. Check drug names.",
            "resolved": rxcui_map,
        }

    interactions = await check_rxnav_interactions(rxcuis)

    return {
        "medications_checked": medications,
        "rxcui_resolved": rxcui_map,
        "interactions": interactions,
        "interaction_count": len(interactions),
        "high_severity_count": sum(1 for i in interactions if i.get("severity", "").upper() in ("HIGH", "CRITICAL")),
        "source": "NLM RxNav Interaction API",
    }


@mcp.tool()
async def get_contraindications(
    drug_name: Annotated[
        str,
        Field(description="Drug name to check. Include dose if known, e.g. 'Levofloxacin 750mg' or just 'Metformin'.")
    ],
    conditions: Annotated[
        Optional[list[str]],
        Field(description="Patient's active ICD-10 condition codes. Example: ['N18.4', 'I50.9', 'J44.1']")
    ] = None,
    allergies: Annotated[
        Optional[list[str]],
        Field(description="Patient's known allergens. Example: ['Penicillin', 'Sulfa', 'NSAIDs']")
    ] = None,
) -> dict:
    """
    Check if a medication has contraindications for a patient's conditions or allergies.
    Checks allergy cross-reactivity (e.g., Amoxicillin in Penicillin-allergic patient)
    and condition contraindications (e.g., Metformin in severe CKD).
    """
    # Convert simple string lists to dicts expected by safety_core
    allergy_dicts = [{"substance": a, "reaction": "unknown", "severity": "unknown"} for a in (allergies or [])]
    condition_dicts = [{"code": c, "display": c} for c in (conditions or [])]

    allergy_contras = check_allergy_cross_reactivity(drug_name, allergy_dicts)
    condition_contras = check_condition_contraindications(drug_name, condition_dicts)

    all_contras = allergy_contras + condition_contras
    has_critical = any(c.get("severity") in ("CRITICAL", "HIGH") for c in all_contras)

    return {
        "drug": drug_name,
        "overall_verdict": "UNSAFE" if has_critical else ("CAUTION" if all_contras else "SAFE"),
        "contraindications": all_contras,
        "contraindication_count": len(all_contras),
        "allergy_contraindications": allergy_contras,
        "condition_contraindications": condition_contras,
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
        Field(description="Patient's ICD-10 condition codes for context. Example: ['J18.9', 'N18.3']")
    ] = None,
    current_medications: Annotated[
        Optional[list[str]],
        Field(description="Patient's current medications to check alternatives against. Example: ['Warfarin 5mg', 'Metformin 850mg']")
    ] = None,
    patient_allergies: Annotated[
        Optional[list[str]],
        Field(description="Known allergens to exclude from alternatives. Example: ['Penicillin', 'Sulfa']")
    ] = None,
) -> dict:
    """
    Suggest safe medication alternatives when a drug has contraindications or interactions.
    Uses clinical LLM reasoning grounded in patient context.
    Alternatives are checked against the patient's current medications and allergies.
    """
    condition_dicts = [{"code": c, "display": c} for c in (patient_conditions or [])]
    med_dicts = [{"drug": m} for m in (current_medications or [])]
    allergy_dicts = [{"substance": a, "reaction": "unknown", "severity": "unknown"} for a in (patient_allergies or [])]

    suggestions = get_alternative_suggestions(
        drug_name=drug_name,
        reason_for_avoidance=reason_for_avoidance,
        indication=indication,
        patient_conditions=condition_dicts,
        current_medications=med_dicts,
        allergies=allergy_dicts,
    )

    if not suggestions:
        return {
            "drug_to_replace": drug_name,
            "alternatives": [],
            "error": "LLM alternative suggestion unavailable. OPENAI_API_KEY not set or API error.",
        }

    return {
        "drug_to_replace": drug_name,
        "reason_for_avoidance": reason_for_avoidance,
        "indication": indication,
        **suggestions,
    }


# ── FastAPI: REST endpoints for MediTwin Orchestrator ─────────────────────────

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


# Create FastAPI for the REST layer
rest_app = FastAPI(
    title="MediTwin Drug Safety Agent",
    description="Drug Safety MCP Server + REST API for MediTwin Orchestrator",
    version="1.0.0",
)


@rest_app.post("/check-safety")
async def check_safety(request: SafetyCheckRequest) -> JSONResponse:
    """
    Full drug safety check — called by MediTwin Orchestrator.
    Combines allergy, condition, and interaction checks in one call.
    """
    result = await _run_full_safety_check(
        proposed_medications=request.proposed_medications,
        current_medications=request.current_medications,
        patient_allergies=request.patient_allergies,
        active_conditions=request.active_conditions,
        patient_id=request.patient_id,
    )
    return JSONResponse(content=result)


@rest_app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={
        "status": "healthy",
        "agent": "drug-safety",
        "version": "1.0.0",
        "mcp_tools": ["check_drug_interactions", "get_contraindications", "suggest_alternatives"],
        "mcp_endpoint": "/mcp/",
        "rest_endpoint": "/check-safety",
        "external_apis": ["NLM RxNav", "NLM RxNorm", "FDA OpenFDA"],
        "local_databases": ["cross_reactivity_table", "contraindications.json"],
    })


# ── Mount MCP + REST into unified ASGI app ───────────────────────────────────
# The MCP streamable-HTTP app runs at /mcp/
# The REST endpoints run at / (root of this app)

from starlette.applications import Starlette
from starlette.routing import Mount, Route

# Get the MCP ASGI app
mcp_asgi = mcp.streamable_http_app()

# Build the combined app using Starlette
combined_app = Starlette(
    routes=[
        Mount("/mcp", app=mcp_asgi),
        Mount("/", app=rest_app),
    ]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="127.0.0.1", port=8004)