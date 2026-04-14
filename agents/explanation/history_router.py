"""
agents/explanation/history_router.py
--------------------------------------
History endpoints for explanation results.
Mirrors agents/diagnosis/history_router.py exactly.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET  /history/{patient_id}              — all explanations for a patient
    GET  /history/{patient_id}/latest       — most recent only
    GET  /history/request/{request_id}      — single record by request ID
    GET  /history/stats/{patient_id}        — aggregate stats for a patient
    DELETE /history/{patient_id}            — delete all records (admin)
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

logger = logging.getLogger("meditwin.explanation.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class ExplanationHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str
    chief_complaint: str
    consensus_status: str
    final_diagnosis: str
    aggregate_confidence: float
    human_review_required: bool
    soap_note: dict
    patient_output: dict
    risk_attribution: dict
    fhir_bundle_summary: dict
    risk_flags: list
    reading_grade_level: float
    reading_acceptable: bool
    reading_attempts: int
    soap_tokens: int
    patient_tokens: int
    source: str
    elapsed_ms: Optional[int]
    cache_hit: bool
    created_at: str


class PatientExplanationHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[ExplanationHistoryItem]


class PatientExplanationStatsResponse(BaseModel):
    patient_id: str
    total_explanations: int
    unique_complaints: int
    human_review_count: int
    avg_confidence: float
    avg_reading_grade: float
    avg_elapsed_ms: Optional[float]
    total_soap_tokens: int
    total_patient_tokens: int
    consensus_breakdown: dict       # {"FULL_CONSENSUS": n, "CONFLICT_RESOLVED": n, ...}
    sources_breakdown: dict         # {"explain": n, "stream": n}
    latest_explanation: Optional[str]
    first_explanation: Optional[str]
    diagnoses_seen: list[dict]      # [{diagnosis, count}] top 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> ExplanationHistoryItem:
    """Normalize asyncpg row → ExplanationHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val

    return ExplanationHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        chief_complaint=row["chief_complaint"],
        consensus_status=row["consensus_status"] or "",
        final_diagnosis=row["final_diagnosis"] or "",
        aggregate_confidence=row["aggregate_confidence"] or 0.0,
        human_review_required=row["human_review_required"] or False,
        soap_note=_load(row["soap_note"]) or {},
        patient_output=_load(row["patient_output"]) or {},
        risk_attribution=_load(row["risk_attribution"]) or {},
        fhir_bundle_summary=_load(row["fhir_bundle_summary"]) or {},
        risk_flags=_load(row["risk_flags"]) or [],
        reading_grade_level=row["reading_grade_level"] or 0.0,
        reading_acceptable=row["reading_acceptable"] or True,
        reading_attempts=row["reading_attempts"] or 1,
        soap_tokens=row["soap_tokens"] or 0,
        patient_tokens=row["patient_tokens"] or 0,
        source=row["source"] or "explain",
        elapsed_ms=row["elapsed_ms"],
        cache_hit=row["cache_hit"] or False,
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientExplanationHistoryResponse)
async def get_patient_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all explanation records for a patient, newest first.
    Supports pagination via limit/offset.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM explanation_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation records found for patient '{patient_id}'",
        )

    return PatientExplanationHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=ExplanationHistoryItem)
async def get_latest_explanation(patient_id: str):
    """Get the most recent explanation for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM explanation_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=ExplanationHistoryItem)
async def get_by_request(request_id: str):
    """Get a single explanation record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=PatientExplanationStatsResponse)
async def get_patient_stats(patient_id: str):
    """
    Aggregate stats for a patient across all explanation sessions.
    Mirrors diagnosis history stats endpoint exactly.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No records found for patient '{patient_id}'",
            )

        unique_complaints = await conn.fetchval(
            "SELECT COUNT(DISTINCT chief_complaint) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        human_review_count = await conn.fetchval(
            "SELECT COUNT(*) FROM explanation_results WHERE patient_id = $1 AND human_review_required = true",
            patient_id,
        )

        avg_confidence = await conn.fetchval(
            "SELECT AVG(aggregate_confidence) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        avg_reading = await conn.fetchval(
            "SELECT AVG(reading_grade_level) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        avg_elapsed = await conn.fetchval(
            "SELECT AVG(elapsed_ms) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        total_soap_tokens = await conn.fetchval(
            "SELECT COALESCE(SUM(soap_tokens), 0) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        total_patient_tokens = await conn.fetchval(
            "SELECT COALESCE(SUM(patient_tokens), 0) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

        consensus_rows = await conn.fetch(
            """
            SELECT consensus_status, COUNT(*) as count
            FROM explanation_results WHERE patient_id = $1
            GROUP BY consensus_status
            """,
            patient_id,
        )

        source_rows = await conn.fetch(
            """
            SELECT source, COUNT(*) as count
            FROM explanation_results WHERE patient_id = $1
            GROUP BY source
            """,
            patient_id,
        )

        dx_rows = await conn.fetch(
            """
            SELECT final_diagnosis, COUNT(*) as count
            FROM explanation_results
            WHERE patient_id = $1 AND final_diagnosis IS NOT NULL AND final_diagnosis != ''
            GROUP BY final_diagnosis
            ORDER BY count DESC
            LIMIT 5
            """,
            patient_id,
        )

        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

    return PatientExplanationStatsResponse(
        patient_id=patient_id,
        total_explanations=total,
        unique_complaints=unique_complaints,
        human_review_count=human_review_count or 0,
        avg_confidence=round(float(avg_confidence or 0), 3),
        avg_reading_grade=round(float(avg_reading or 0), 1),
        avg_elapsed_ms=round(float(avg_elapsed), 1) if avg_elapsed else None,
        total_soap_tokens=int(total_soap_tokens or 0),
        total_patient_tokens=int(total_patient_tokens or 0),
        consensus_breakdown={r["consensus_status"]: r["count"] for r in consensus_rows},
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_explanation=str(latest) if latest else None,
        first_explanation=str(first) if first else None,
        diagnoses_seen=[
            {"diagnosis": r["final_diagnosis"], "count": r["count"]}
            for r in dx_rows
        ],
    )


@router.delete("/{patient_id}")
async def delete_patient_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete all explanation records for a patient.
    Requires internal token — admin only.
    """
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM explanation_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} explanation records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}