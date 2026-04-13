"""
agents/drug_safety/history_router.py
--------------------------------------
History endpoints for drug safety check results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}              — all safety checks for a patient
    GET /history/{patient_id}/latest       — most recent check only
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

logger = logging.getLogger("meditwin.drug_safety.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class DrugSafetyHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str

    proposed_medications: list
    current_medications: list
    proposed_count: int
    current_count: int

    safety_status: str

    contraindications: list
    contraindication_count: int
    approved_medications: list
    flagged_medications: list
    approved_count: int
    flagged_count: int

    critical_interactions: list
    interaction_count: int
    fda_warnings: Optional[dict]
    black_box_count: int

    patient_risk_profile: Optional[dict]
    interaction_risk_narrative: Optional[str]
    overall_risk_level: Optional[str]
    recommended_action: Optional[str]
    llm_enriched: bool

    fhir_medication_requests: list

    cache_hit: bool
    elapsed_ms: Optional[int]
    source: str
    created_at: str


class DrugSafetyHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[DrugSafetyHistoryItem]


class DrugSafetyStatsResponse(BaseModel):
    patient_id: str
    total_checks: int

    # Verdict breakdown
    safe_count: int
    caution_count: int
    unsafe_count: int

    # Interaction/contraindication totals
    total_contraindications: int
    total_interactions: int
    total_black_box_warnings: int

    # LLM + risk breakdown
    llm_enriched_count: int
    risk_level_breakdown: dict     # {CRITICAL: n, HIGH: n, MODERATE: n, LOW: n, MINIMAL: n}

    # Drug-specific insight
    most_flagged_drugs: list[dict]   # [{drug, flag_count}] top 5

    # Source breakdown
    sources_breakdown: dict

    # Timeline
    latest_check: Optional[str]
    first_check: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> DrugSafetyHistoryItem:
    """Normalize asyncpg row → DrugSafetyHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val  # already dict/list from asyncpg JSONB

    return DrugSafetyHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        proposed_medications=_load(row["proposed_medications"]) or [],
        current_medications=_load(row["current_medications"]) or [],
        proposed_count=row["proposed_count"] or 0,
        current_count=row["current_count"] or 0,
        safety_status=row["safety_status"] or "UNKNOWN",
        contraindications=_load(row["contraindications"]) or [],
        contraindication_count=row["contraindication_count"] or 0,
        approved_medications=_load(row["approved_medications"]) or [],
        flagged_medications=_load(row["flagged_medications"]) or [],
        approved_count=row["approved_count"] or 0,
        flagged_count=row["flagged_count"] or 0,
        critical_interactions=_load(row["critical_interactions"]) or [],
        interaction_count=row["interaction_count"] or 0,
        fda_warnings=_load(row["fda_warnings"]),
        black_box_count=row["black_box_count"] or 0,
        patient_risk_profile=_load(row["patient_risk_profile"]),
        interaction_risk_narrative=row["interaction_risk_narrative"],
        overall_risk_level=row["overall_risk_level"],
        recommended_action=row["recommended_action"],
        llm_enriched=row["llm_enriched"] or False,
        fhir_medication_requests=_load(row["fhir_medication_requests"]) or [],
        cache_hit=row["cache_hit"] or False,
        elapsed_ms=row["elapsed_ms"],
        source=row["source"] or "check-safety",
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=DrugSafetyHistoryResponse)
async def get_patient_drug_safety_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all drug safety check records for a patient, newest first.
    Supports pagination via limit/offset.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM drug_safety_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM drug_safety_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No drug safety records found for patient '{patient_id}'",
        )

    return DrugSafetyHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=DrugSafetyHistoryItem)
async def get_latest_drug_safety(patient_id: str):
    """Get the most recent drug safety check for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM drug_safety_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No drug safety records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=DrugSafetyHistoryItem)
async def get_by_request(request_id: str):
    """Get a single drug safety record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=DrugSafetyStatsResponse)
async def get_patient_drug_safety_stats(patient_id: str):
    """
    Aggregate stats for a patient across all drug safety check sessions.
    Useful for frontend patient summary cards and prescriber dashboards.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM drug_safety_results WHERE patient_id = $1",
            patient_id,
        )

        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No drug safety records found for patient '{patient_id}'",
            )

        # Verdict breakdown
        verdict_rows = await conn.fetch(
            """
            SELECT safety_status, COUNT(*) AS count
            FROM drug_safety_results
            WHERE patient_id = $1
            GROUP BY safety_status
            """,
            patient_id,
        )
        verdict_map = {r["safety_status"]: r["count"] for r in verdict_rows}

        # Totals across all sessions
        totals_row = await conn.fetchrow(
            """
            SELECT
                SUM(contraindication_count) AS total_contras,
                SUM(interaction_count)      AS total_interactions,
                SUM(black_box_count)        AS total_black_box
            FROM drug_safety_results
            WHERE patient_id = $1
            """,
            patient_id,
        )

        # LLM enrichment count
        llm_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM drug_safety_results
            WHERE patient_id = $1 AND llm_enriched = TRUE
            """,
            patient_id,
        )

        # Risk level breakdown
        risk_rows = await conn.fetch(
            """
            SELECT overall_risk_level, COUNT(*) AS count
            FROM drug_safety_results
            WHERE patient_id = $1 AND overall_risk_level IS NOT NULL
            GROUP BY overall_risk_level
            """,
            patient_id,
        )

        # Most flagged drugs — unnest flagged_medications JSONB array
        flagged_rows = await conn.fetch(
            """
            SELECT drug, COUNT(*) AS flag_count
            FROM (
                SELECT jsonb_array_elements_text(flagged_medications) AS drug
                FROM drug_safety_results
                WHERE patient_id = $1
                  AND jsonb_array_length(flagged_medications) > 0
            ) sub
            GROUP BY drug
            ORDER BY flag_count DESC
            LIMIT 5
            """,
            patient_id,
        )

        # Source breakdown
        source_rows = await conn.fetch(
            """
            SELECT source, COUNT(*) AS count
            FROM drug_safety_results
            WHERE patient_id = $1
            GROUP BY source
            """,
            patient_id,
        )

        # Timeline
        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM drug_safety_results WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM drug_safety_results WHERE patient_id = $1",
            patient_id,
        )

    return DrugSafetyStatsResponse(
        patient_id=patient_id,
        total_checks=total,
        safe_count=verdict_map.get("SAFE", 0),
        caution_count=verdict_map.get("CAUTION", 0),
        unsafe_count=verdict_map.get("UNSAFE", 0),
        total_contraindications=int(totals_row["total_contras"] or 0),
        total_interactions=int(totals_row["total_interactions"] or 0),
        total_black_box_warnings=int(totals_row["total_black_box"] or 0),
        llm_enriched_count=llm_count or 0,
        risk_level_breakdown={r["overall_risk_level"]: r["count"] for r in risk_rows},
        most_flagged_drugs=[
            {"drug": r["drug"], "flag_count": r["flag_count"]}
            for r in flagged_rows
        ],
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_check=str(latest) if latest else None,
        first_check=str(first) if first else None,
    )


@router.delete("/{patient_id}")
async def delete_patient_drug_safety_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete all drug safety records for a patient.
    Requires internal token — admin only.
    """
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM drug_safety_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} drug safety records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}