"""
agents/patient_context/stream_endpoint.py
------------------------------------------
SSE streaming endpoint for the Patient Context Agent.

ADD this to agents/patient_context/main.py:

    from stream_endpoint import router as stream_router
    app.include_router(stream_router)

The /stream endpoint does the exact same work as /fetch but
emits SSE events so the orchestrator can proxy real-time
progress to the frontend.

Events emitted (in order):
  1. status  — "Connecting to FHIR server..."
  2. status  — "Fetching patient resources in parallel..."
  3. progress — per resource type fetched (6 total)
  4. status  — "Normalizing FHIR resources..."
  5. complete — full PatientState + cache/timing metadata
  | error    — if patient not found or FHIR unreachable
"""

from __future__ import annotations

import asyncio
import sys, os
import time
from typing import Optional, AsyncIterator

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, sse_done,
)

router = APIRouter()


# ── We reuse the logic already in main.py ─────────────────────────────────────
# Import helpers from the main module (works because main.py adds itself to path)
# These are the same functions used by /fetch — no duplication.

from utils import (
    fetch_fhir_resource,
    normalize_patient,
    normalize_conditions,
    normalize_medications,
    normalize_allergies,
    normalize_observations,
    normalize_diagnostic_reports,
)


from shared.redis_client import redis_client
from shared.models import PatientState
from datetime import datetime
from pydantic import ValidationError

# at the top of stream_endpoint.py, add to existing imports:
import uuid
from db import save_patient_context, PatientContextRecord


# ── Request body (mirrored from main.py) ──────────────────────────────────────

from pydantic import BaseModel

class StreamFetchRequest(BaseModel):
    patient_id: Optional[str] = None
    fhir_base_url: Optional[str] = "https://hapi.fhir.org/baseR4"
    sharp_token: Optional[str] = None


# ── Core async generator ───────────────────────────────────────────────────────

async def _stream_fetch(
    patient_id: str,
    fhir_base_url: str,
    sharp_token: Optional[str],
    source: str,
) -> AsyncIterator[str]:
    """Yields SSE strings while fetching + normalizing patient data."""

    timer = Timer()
    node = "patient_context"

    yield evt_status(node, f"Patient Context Agent starting for patient {patient_id}",
                     step=1, total=4)

    # ── Redis cache check ──────────────────────────────────────────────────────
    cache_key = f"patient_state:{patient_id}"
    cached = await redis_client.get_json(cache_key)

    if cached:
        elapsed = timer.elapsed_ms()
        yield evt_status(node, "Cache hit — serving patient data from Redis")
        yield evt_complete(node, {
            "patient_state": cached,
            "cache_hit": True,
            "fetch_time_ms": elapsed,
            "source": source,
            "fhir_resources_fetched": 0,
        }, elapsed_ms=elapsed)
        return

    # ── FHIR parallel fetch ────────────────────────────────────────────────────
    auth_headers = {}
    if sharp_token:
        tok = sharp_token if sharp_token.startswith("Bearer ") else f"Bearer {sharp_token}"
        auth_headers["Authorization"] = tok

    yield evt_status(node,
                     "Fetching 6 FHIR resource types in parallel...",
                     step=2, total=4)

    # Fire all 6 requests concurrently
    resource_names = [
        "Patient", "Condition", "MedicationRequest",
        "AllergyIntolerance", "Observation", "DiagnosticReport",
    ]

    tasks = [
        fetch_fhir_resource("Patient",              fhir_base_url,
                             resource_id=patient_id, auth_headers=auth_headers),
        fetch_fhir_resource("Condition",            fhir_base_url,
                             search_params={"patient": patient_id, "clinical-status": "active"},
                             auth_headers=auth_headers),
        fetch_fhir_resource("MedicationRequest",    fhir_base_url,
                             search_params={"patient": patient_id, "status": "active"},
                             auth_headers=auth_headers),
        fetch_fhir_resource("AllergyIntolerance",   fhir_base_url,
                             search_params={"patient": patient_id},
                             auth_headers=auth_headers),
        fetch_fhir_resource("Observation",          fhir_base_url,
                             search_params={"patient": patient_id, "category": "laboratory",
                                            "_sort": "-date", "_count": "20"},
                             auth_headers=auth_headers),
        fetch_fhir_resource("DiagnosticReport",     fhir_base_url,
                             search_params={"patient": patient_id,
                                            "_sort": "-date", "_count": "5"},
                             auth_headers=auth_headers),
    ]

    results = await asyncio.gather(*tasks)

    # Emit a progress tick per resource
    for i, (name, res) in enumerate(zip(resource_names, results), 1):
        found = bool(res)
        yield evt_progress(node,
                           f"Fetched {name} — {'✓' if found else 'empty'}",
                           pct=round(i / len(resource_names) * 60, 1))  # 0-60%

    (patient_data, conditions_bundle, medications_bundle,
     allergies_bundle, observations_bundle, diagnostic_reports_bundle) = results

    # ── Normalization ──────────────────────────────────────────────────────────
    yield evt_status(node, "Normalizing FHIR resources into PatientState...",
                     step=3, total=4)

    demographics = normalize_patient(patient_data)
    if demographics is None:
        yield evt_error(node,
                        f"Patient '{patient_id}' not found or invalid Patient resource.",
                        fatal=True)
        return

    elapsed = timer.elapsed_ms()

    try:
        conditions              = normalize_conditions(conditions_bundle)
        medications             = await normalize_medications(medications_bundle, base_url=fhir_base_url, auth_headers=auth_headers)
        allergies               = normalize_allergies(allergies_bundle)
        labs                    = normalize_observations(observations_bundle)
        diagnostic_reports, img = normalize_diagnostic_reports(diagnostic_reports_bundle)

    # ── Persist to DB (non-fatal — mirrors /fetch) ─────────────────────────────
        request_id = str(uuid.uuid4())[:8]
        await save_patient_context(PatientContextRecord(
            request_id=request_id,
            patient_id=patient_id,
            source=source,
            fhir_base_url=fhir_base_url,
            fhir_resources_fetched=6,
            demographics=demographics.model_dump() if demographics else None,
            active_conditions=[c.model_dump() for c in conditions],
            medications=[m.model_dump() for m in medications],
            allergies=[a.model_dump() for a in allergies],
            lab_results=[l.model_dump() for l in labs],
            diagnostic_reports=[d.model_dump() for d in diagnostic_reports],
            imaging_available=img,
            cache_hit=False,
            fetch_time_ms=elapsed,
        ))

        patient_state = PatientState(
            patient_id=patient_id,
            demographics=demographics,
            active_conditions=conditions,
            medications=medications,
            allergies=allergies,
            lab_results=labs,
            diagnostic_reports=diagnostic_reports,
            recent_encounters=[],
            state_timestamp=datetime.now().isoformat() + "Z",
            imaging_available=img,
        )
    except (ValidationError, Exception) as exc:
        yield evt_error(node, f"Normalization failed: {exc}", fatal=True)
        return

    # ── Cache ──────────────────────────────────────────────────────────────────
    await redis_client.set_json(cache_key, patient_state.model_dump(), ttl=600)

    yield evt_status(node, "Patient data ready ✓", step=4, total=4)
    yield evt_complete(node, {
        "patient_state": patient_state.model_dump(),
        "cache_hit": False,
        "fetch_time_ms": elapsed,
        "source": source,
        "fhir_resources_fetched": 6,
        "summary": {
            "name":        demographics.name,
            "age":         demographics.age,
            "conditions":  len(conditions),
            "medications": len(medications),
            "labs":        len(labs),
            "allergies":   len(allergies),
        },
    }, elapsed_ms=elapsed)


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@router.post("/stream")
async def stream_fetch(
    request: StreamFetchRequest,
    x_sharp_patient_id:    Optional[str] = Header(None, alias="X-SHARP-Patient-ID"),
    x_sharp_fhir_token:    Optional[str] = Header(None, alias="X-SHARP-FHIR-Token"),
    x_sharp_fhir_base_url: Optional[str] = Header(None, alias="X-SHARP-FHIR-BaseURL"),
):
    """
    SSE streaming version of /fetch.
    Emits status/progress/complete events as the patient data is loaded.
    """
    patient_id    = x_sharp_patient_id    or request.patient_id
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"
    sharp_token   = x_sharp_fhir_token    or request.sharp_token
    source        = "SHARP" if x_sharp_patient_id else "direct"

    async def generator():
        if not patient_id:
            yield evt_error("patient_context",
                            "patient_id is required (via header or body)",
                            fatal=True)
            yield sse_done()
            return

        async for chunk in _stream_fetch(patient_id, fhir_base_url, sharp_token, source):
            yield chunk

        yield sse_done()

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers=SSE_HEADERS)