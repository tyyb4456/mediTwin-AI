"""
agents/lab_analysis/history_router.py
---------------------------------------
History endpoints for lab analysis results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}                  — all lab results for a patient
    GET /history/{patient_id}/latest           — most recent lab result only
    GET /history/request/{request_id}          — single record by request ID
    GET /history/stats/{patient_id}            — aggregate stats for a patient
    DELETE /history/{patient_id}               — delete all records for a patient (admin)
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

logger = logging.getLogger("meditwin.lab_analysis.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class LabHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str
    total_results: int
    abnormal_count: int
    critical_count: int
    overall_severity: str
    severity_score: Optional[dict]
    flagged_results: list
    identified_patterns: list
    critical_alerts: list
    confirms_top_diagnosis: bool
    proposed_diagnosis: str
    proposed_icd10: str
    lab_confidence_boost: float
    alternative_diagnosis_code: Optional[str]
    clinical_decision_support: Optional[dict]
    trend_analysis: Optional[list]
    llm_interpretation_available: bool
    cache_hit: bool
    elapsed_ms: Optional[int]
    source: str
    created_at: str


class PatientLabHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[LabHistoryItem]


class PatientLabStatsResponse(BaseModel):
    patient_id: str
    total_lab_sessions: int
    total_abnormal_findings: int
    total_critical_findings: int
    severity_breakdown: dict          # {"CRITICAL": n, "HIGH": n, "MODERATE": n, "NORMAL": n}
    top_icd10_confirmed: list[dict]   # [{icd10, display, count}] — confirmed diagnoses
    critical_alert_sessions: int      # sessions with at least one critical alert
    llm_available_sessions: int
    sources_breakdown: dict           # {"analyze-labs": n, "stream": n}
    latest_session: Optional[str]
    first_session: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> LabHistoryItem:
    """Normalize asyncpg row → LabHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val  # already dict/list from asyncpg JSONB

    return LabHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        total_results=row["total_results"] or 0,
        abnormal_count=row["abnormal_count"] or 0,
        critical_count=row["critical_count"] or 0,
        overall_severity=row["overall_severity"] or "NORMAL",
        severity_score=_load(row["severity_score"]),
        flagged_results=_load(row["flagged_results"]) or [],
        identified_patterns=_load(row["identified_patterns"]) or [],
        critical_alerts=_load(row["critical_alerts"]) or [],
        confirms_top_diagnosis=row["confirms_top_diagnosis"] or False,
        proposed_diagnosis=row["proposed_diagnosis"] or "",
        proposed_icd10=row["proposed_icd10"] or "",
        lab_confidence_boost=float(row["lab_confidence_boost"] or 0.0),
        alternative_diagnosis_code=row["alternative_diagnosis_code"],
        clinical_decision_support=_load(row["clinical_decision_support"]),
        trend_analysis=_load(row["trend_analysis"]),
        llm_interpretation_available=row["llm_interpretation_available"] or False,
        cache_hit=row["cache_hit"] or False,
        elapsed_ms=row["elapsed_ms"],
        source=row["source"] or "analyze-labs",
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientLabHistoryResponse)
async def get_patient_lab_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all lab analysis records for a patient, newest first.
    Supports pagination via limit/offset.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM lab_analysis_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM lab_analysis_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No lab analysis records found for patient '{patient_id}'",
        )

    return PatientLabHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=LabHistoryItem)
async def get_latest_lab_result(patient_id: str):
    """Get the most recent lab analysis for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM lab_analysis_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No lab analysis records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=LabHistoryItem)
async def get_by_request(request_id: str):
    """Get a single lab analysis record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=PatientLabStatsResponse)
async def get_patient_lab_stats(patient_id: str):
    """
    Aggregate stats for a patient across all lab analysis sessions.
    Useful for frontend patient summary cards and trend dashboards.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM lab_analysis_results WHERE patient_id = $1",
            patient_id,
        )

        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No lab records found for patient '{patient_id}'",
            )

        # Total abnormal and critical across all sessions
        totals_row = await conn.fetchrow(
            """
            SELECT
                SUM(abnormal_count) AS total_abnormal,
                SUM(critical_count) AS total_critical
            FROM lab_analysis_results
            WHERE patient_id = $1
            """,
            patient_id,
        )

        # Severity breakdown
        severity_rows = await conn.fetch(
            """
            SELECT overall_severity, COUNT(*) AS count
            FROM lab_analysis_results
            WHERE patient_id = $1
            GROUP BY overall_severity
            """,
            patient_id,
        )

        # Top confirmed ICD-10 codes
        top_icd10_rows = await conn.fetch(
            """
            SELECT proposed_icd10, proposed_diagnosis, COUNT(*) AS count
            FROM lab_analysis_results
            WHERE patient_id = $1 AND confirms_top_diagnosis = TRUE
            GROUP BY proposed_icd10, proposed_diagnosis
            ORDER BY count DESC
            LIMIT 5
            """,
            patient_id,
        )

        # Sessions with at least one critical alert (non-empty array)
        critical_sessions = await conn.fetchval(
            """
            SELECT COUNT(*) FROM lab_analysis_results
            WHERE patient_id = $1
              AND jsonb_array_length(critical_alerts) > 0
            """,
            patient_id,
        )

        # Sessions where LLM was available
        llm_sessions = await conn.fetchval(
            """
            SELECT COUNT(*) FROM lab_analysis_results
            WHERE patient_id = $1 AND llm_interpretation_available = TRUE
            """,
            patient_id,
        )

        # Source breakdown
        source_rows = await conn.fetch(
            """
            SELECT source, COUNT(*) AS count
            FROM lab_analysis_results
            WHERE patient_id = $1
            GROUP BY source
            """,
            patient_id,
        )

        # Timeline
        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM lab_analysis_results WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM lab_analysis_results WHERE patient_id = $1",
            patient_id,
        )

    return PatientLabStatsResponse(
        patient_id=patient_id,
        total_lab_sessions=total,
        total_abnormal_findings=int(totals_row["total_abnormal"] or 0),
        total_critical_findings=int(totals_row["total_critical"] or 0),
        severity_breakdown={r["overall_severity"]: r["count"] for r in severity_rows},
        top_icd10_confirmed=[
            {"icd10": r["proposed_icd10"], "display": r["proposed_diagnosis"], "count": r["count"]}
            for r in top_icd10_rows
        ],
        critical_alert_sessions=critical_sessions or 0,
        llm_available_sessions=llm_sessions or 0,
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_session=str(latest) if latest else None,
        first_session=str(first) if first else None,
    )


@router.delete("/{patient_id}")
async def delete_patient_lab_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete all lab analysis records for a patient.
    Requires internal token — admin only.
    """
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM lab_analysis_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} lab records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}