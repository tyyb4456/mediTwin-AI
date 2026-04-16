"""
agents/patient_context/history_router.py
------------------------------------------
History endpoints for patient context fetch results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}              — all fetch records for a patient
    GET /history/{patient_id}/latest       — most recent fetch only
    GET /history/request/{request_id}      — single record by request ID
    GET /history/stats/{patient_id}        — aggregate stats for a patient
    DELETE /history/{patient_id}           — delete all records for a patient (admin)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel

import db
from db import get_by_patient, get_by_request_id, is_available

logger = logging.getLogger("meditwin.patient_context.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class PatientContextHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str
    source: str
    fhir_base_url: Optional[str]
    fhir_resources_fetched: int
    demographics: Optional[dict]
    active_conditions: list
    medications: list
    allergies: list
    lab_results: list
    diagnostic_reports: list
    imaging_available: bool
    conditions_count: int
    medications_count: int
    allergies_count: int
    lab_results_count: int
    diagnostic_reports_count: int
    cache_hit: bool
    fetch_time_ms: Optional[int]
    created_at: str


class PatientContextHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[PatientContextHistoryItem]


class PatientContextStatsResponse(BaseModel):
    patient_id: str
    total_fetch_sessions: int
    cache_hit_sessions: int
    cache_miss_sessions: int
    sharp_sessions: int               # fetches via SHARP headers
    direct_sessions: int              # fetches via request body
    imaging_available_sessions: int
    avg_fetch_time_ms: Optional[float]
    avg_conditions_count: Optional[float]
    avg_medications_count: Optional[float]
    avg_lab_results_count: Optional[float]
    peak_conditions_count: int
    peak_lab_results_count: int
    latest_fetch: Optional[str]
    first_fetch: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> PatientContextHistoryItem:
    """Normalize asyncpg row → PatientContextHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val  # already dict/list from asyncpg JSONB

    return PatientContextHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        source=row["source"] or "direct",
        fhir_base_url=row["fhir_base_url"],
        fhir_resources_fetched=row["fhir_resources_fetched"] or 0,
        demographics=_load(row["demographics"]),
        active_conditions=_load(row["active_conditions"]) or [],
        medications=_load(row["medications"]) or [],
        allergies=_load(row["allergies"]) or [],
        lab_results=_load(row["lab_results"]) or [],
        diagnostic_reports=_load(row["diagnostic_reports"]) or [],
        imaging_available=row["imaging_available"] or False,
        conditions_count=row["conditions_count"] or 0,
        medications_count=row["medications_count"] or 0,
        allergies_count=row["allergies_count"] or 0,
        lab_results_count=row["lab_results_count"] or 0,
        diagnostic_reports_count=row["diagnostic_reports_count"] or 0,
        cache_hit=row["cache_hit"] or False,
        fetch_time_ms=row["fetch_time_ms"],
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientContextHistoryResponse)
async def get_patient_context_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all patient context fetch records, newest first.
    Supports pagination via limit/offset.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM patient_context_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM patient_context_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No patient context records found for patient '{patient_id}'",
        )

    return PatientContextHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=PatientContextHistoryItem)
async def get_latest_patient_context(patient_id: str):
    """Get the most recent patient context fetch for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM patient_context_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No patient context records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=PatientContextHistoryItem)
async def get_by_request(request_id: str):
    """Get a single patient context record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)

@router.delete("/{patient_id}")
async def delete_patient_context_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete all patient context records for a patient.
    Requires internal token — admin only.
    """
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM patient_context_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} patient context records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}