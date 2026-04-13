"""
agents/imaging_triage/history_router.py
-----------------------------------------
History endpoints for imaging triage results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}              — all imaging records for a patient
    GET /history/{patient_id}/latest       — most recent record only
    GET /history/request/{request_id}      — single record by request ID
    GET /history/stats/{patient_id}        — aggregate stats for a patient
    DELETE /history/{patient_id}           — delete all records (admin)
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

logger = logging.getLogger("meditwin.imaging_triage.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class ImagingHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str

    analysis_mode: str
    model_loaded: bool
    image_provided: bool

    prediction: Optional[str]
    confidence: Optional[float]
    pneumonia_probability: Optional[float]
    normal_probability: Optional[float]

    triage_grade: Optional[str]
    triage_priority: Optional[int]
    triage_label: Optional[str]

    pattern: Optional[str]
    affected_area: Optional[str]
    bilateral: bool
    confidence_in_findings: Optional[str]

    clinical_interpretation: Optional[str]
    confirms_diagnosis: bool
    diagnosis_code: Optional[str]

    patient_age: Optional[int]
    patient_gender: Optional[str]
    chief_complaint: Optional[str]

    llm_enriched: bool
    llm_token_count: int
    synthetic_reasoning: Optional[str]

    fhir_diagnostic_report: Optional[dict]

    mock: bool
    cache_hit: bool
    elapsed_ms: Optional[int]
    source: str
    created_at: str


class ImagingHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[ImagingHistoryItem]


class ImagingStatsResponse(BaseModel):
    patient_id: str
    total_scans: int

    # Mode breakdown
    cnn_scans: int
    synthetic_scans: int
    mock_scans: int

    # Prediction breakdown
    pneumonia_detected: int
    normal_detected: int
    unknown_prediction: int

    # Triage breakdown
    triage_breakdown: dict       # {IMMEDIATE: n, URGENT: n, SEMI-URGENT: n, ROUTINE: n}
    grade_breakdown: dict        # {SEVERE: n, MODERATE: n, MILD: n, NORMAL: n}

    # Clinical confirmations
    diagnosis_confirmed_count: int

    # LLM usage
    llm_enriched_count: int
    avg_llm_tokens: Optional[float]

    # Source breakdown
    sources_breakdown: dict

    # Timeline
    latest_scan: Optional[str]
    first_scan: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> ImagingHistoryItem:
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val

    return ImagingHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        analysis_mode=row["analysis_mode"] or "cnn",
        model_loaded=row["model_loaded"] or False,
        image_provided=row["image_provided"] or False,
        prediction=row["prediction"],
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        pneumonia_probability=float(row["pneumonia_probability"]) if row["pneumonia_probability"] is not None else None,
        normal_probability=float(row["normal_probability"]) if row["normal_probability"] is not None else None,
        triage_grade=row["triage_grade"],
        triage_priority=row["triage_priority"],
        triage_label=row["triage_label"],
        pattern=row["pattern"],
        affected_area=row["affected_area"],
        bilateral=row["bilateral"] or False,
        confidence_in_findings=row["confidence_in_findings"],
        clinical_interpretation=row["clinical_interpretation"],
        confirms_diagnosis=row["confirms_diagnosis"] or False,
        diagnosis_code=row["diagnosis_code"],
        patient_age=row["patient_age"],
        patient_gender=row["patient_gender"],
        chief_complaint=row["chief_complaint"],
        llm_enriched=row["llm_enriched"] or False,
        llm_token_count=row["llm_token_count"] or 0,
        synthetic_reasoning=row["synthetic_reasoning"],
        fhir_diagnostic_report=_load(row["fhir_diagnostic_report"]),
        mock=row["mock"] or False,
        cache_hit=row["cache_hit"] or False,
        elapsed_ms=row["elapsed_ms"],
        source=row["source"] or "analyze-xray",
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=ImagingHistoryResponse)
async def get_patient_imaging_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """All imaging triage records for a patient, newest first."""
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM imaging_triage_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM imaging_triage_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No imaging records found for patient '{patient_id}'",
        )

    return ImagingHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=ImagingHistoryItem)
async def get_latest_imaging(patient_id: str):
    """Most recent imaging triage record for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM imaging_triage_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No imaging records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=ImagingHistoryItem)
async def get_by_request(request_id: str):
    """Single imaging record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=ImagingStatsResponse)
async def get_patient_imaging_stats(patient_id: str):
    """Aggregate imaging stats for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM imaging_triage_results WHERE patient_id = $1",
            patient_id,
        )
        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No imaging records found for patient '{patient_id}'",
            )

        # Mode breakdown
        mode_rows = await conn.fetch(
            """SELECT analysis_mode, COUNT(*) AS count
               FROM imaging_triage_results WHERE patient_id = $1
               GROUP BY analysis_mode""",
            patient_id,
        )
        mode_map = {r["analysis_mode"]: r["count"] for r in mode_rows}

        # Prediction breakdown
        pred_rows = await conn.fetch(
            """SELECT prediction, COUNT(*) AS count
               FROM imaging_triage_results WHERE patient_id = $1
               GROUP BY prediction""",
            patient_id,
        )
        pred_map = {r["prediction"]: r["count"] for r in pred_rows if r["prediction"]}

        # Triage label breakdown
        triage_rows = await conn.fetch(
            """SELECT triage_label, COUNT(*) AS count
               FROM imaging_triage_results WHERE patient_id = $1 AND triage_label IS NOT NULL
               GROUP BY triage_label""",
            patient_id,
        )

        # Grade breakdown
        grade_rows = await conn.fetch(
            """SELECT triage_grade, COUNT(*) AS count
               FROM imaging_triage_results WHERE patient_id = $1 AND triage_grade IS NOT NULL
               GROUP BY triage_grade""",
            patient_id,
        )

        # Diagnosis confirmed count
        confirmed = await conn.fetchval(
            """SELECT COUNT(*) FROM imaging_triage_results
               WHERE patient_id = $1 AND confirms_diagnosis = TRUE""",
            patient_id,
        )

        # LLM stats
        llm_row = await conn.fetchrow(
            """SELECT COUNT(*) AS enriched_count, AVG(llm_token_count) AS avg_tokens
               FROM imaging_triage_results
               WHERE patient_id = $1 AND llm_enriched = TRUE""",
            patient_id,
        )

        # Source breakdown
        source_rows = await conn.fetch(
            """SELECT source, COUNT(*) AS count
               FROM imaging_triage_results WHERE patient_id = $1
               GROUP BY source""",
            patient_id,
        )

        # Timeline
        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM imaging_triage_results WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM imaging_triage_results WHERE patient_id = $1",
            patient_id,
        )

    return ImagingStatsResponse(
        patient_id=patient_id,
        total_scans=total,
        cnn_scans=mode_map.get("cnn", 0),
        synthetic_scans=mode_map.get("synthetic", 0),
        mock_scans=mode_map.get("mock", 0),
        pneumonia_detected=pred_map.get("PNEUMONIA", 0),
        normal_detected=pred_map.get("NORMAL", 0),
        unknown_prediction=pred_map.get("MOCK_NO_MODEL", 0) + pred_map.get("SYNTHETIC", 0),
        triage_breakdown={r["triage_label"]: r["count"] for r in triage_rows},
        grade_breakdown={r["triage_grade"]: r["count"] for r in grade_rows},
        diagnosis_confirmed_count=confirmed or 0,
        llm_enriched_count=llm_row["enriched_count"] or 0,
        avg_llm_tokens=round(float(llm_row["avg_tokens"]), 1) if llm_row["avg_tokens"] else None,
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_scan=str(latest) if latest else None,
        first_scan=str(first) if first else None,
    )


@router.delete("/{patient_id}")
async def delete_patient_imaging_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """Delete all imaging records for a patient. Admin only."""
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM imaging_triage_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} imaging records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}