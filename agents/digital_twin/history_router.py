"""
agents/digital_twin/history_router.py
---------------------------------------
History endpoints for digital twin simulation results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}                  — all simulations for a patient
    GET /history/{patient_id}/latest           — most recent simulation only
    GET /history/request/{request_id}          — single record by request ID
    GET /history/stats/{patient_id}            — aggregate stats for a patient
    GET /history/compare/{patient_id}          — compare outcomes across simulations
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

logger = logging.getLogger("meditwin.digital_twin.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class SimulationHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str
    diagnosis: str
    diagnosis_code: Optional[str]
    patient_risk_profile: str
    baseline_mortality_30d: float
    baseline_readmission_30d: float
    baseline_complication: float
    recommended_option: str
    recommendation_confidence: float
    model_confidence: str
    treatment_options_count: int
    scenarios: list
    simulation_summary: dict
    what_if_narrative: str
    fhir_care_plan: Optional[dict]
    feature_attribution: list
    sensitivity_analysis: Optional[list]
    cost_effectiveness: Optional[dict]
    models_loaded: bool
    cache_hit: bool
    elapsed_ms: Optional[int]
    source: str
    created_at: str


class PatientSimulationHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[SimulationHistoryItem]


class PatientSimulationStatsResponse(BaseModel):
    patient_id: str
    total_simulations: int
    unique_diagnoses: int
    risk_profile_breakdown: dict
    recommendation_trends: dict
    average_mortality_risk: float
    average_readmission_risk: float
    average_recommendation_confidence: float
    model_confidence_breakdown: dict
    most_common_diagnosis: Optional[str]
    most_recommended_option: Optional[str]
    sources_breakdown: dict
    latest_simulation: Optional[str]
    first_simulation: Optional[str]


class ComparisonResponse(BaseModel):
    patient_id: str
    total_simulations: int
    comparison_summary: dict
    risk_trajectories: dict
    treatment_effectiveness: dict


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> SimulationHistoryItem:
    """Normalize asyncpg row → SimulationHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val

    return SimulationHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        diagnosis=row["diagnosis"],
        diagnosis_code=row["diagnosis_code"],
        patient_risk_profile=row["patient_risk_profile"] or "UNKNOWN",
        baseline_mortality_30d=float(row["baseline_mortality_30d"] or 0.0),
        baseline_readmission_30d=float(row["baseline_readmission_30d"] or 0.0),
        baseline_complication=float(row["baseline_complication"] or 0.0),
        recommended_option=row["recommended_option"],
        recommendation_confidence=float(row["recommendation_confidence"] or 0.0),
        model_confidence=row["model_confidence"] or "UNKNOWN",
        treatment_options_count=row["treatment_options_count"] or 0,
        scenarios=_load(row["scenarios"]) or [],
        simulation_summary=_load(row["simulation_summary"]) or {},
        what_if_narrative=row["what_if_narrative"] or "",
        fhir_care_plan=_load(row["fhir_care_plan"]),
        feature_attribution=_load(row["feature_attribution"]) or [],
        sensitivity_analysis=_load(row["sensitivity_analysis"]),
        cost_effectiveness=_load(row["cost_effectiveness"]),
        models_loaded=row["models_loaded"] or False,
        cache_hit=row["cache_hit"] or False,
        elapsed_ms=row["elapsed_ms"],
        source=row["source"] or "simulate",
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientSimulationHistoryResponse)
async def get_patient_simulation_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Get all simulation records for a patient, newest first."""
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM digital_twin_simulations
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation records found for patient '{patient_id}'",
        )

    return PatientSimulationHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=SimulationHistoryItem)
async def get_latest_simulation(patient_id: str):
    """Get the most recent digital twin simulation for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM digital_twin_simulations
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=SimulationHistoryItem)
async def get_by_request(request_id: str):
    """Get a single simulation record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=PatientSimulationStatsResponse)
async def get_patient_simulation_stats(patient_id: str):
    """
    Aggregate stats for a patient across all digital twin simulations.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )

        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No simulation records found for patient '{patient_id}'",
            )

        unique_dx = await conn.fetchval(
            "SELECT COUNT(DISTINCT diagnosis_code) FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )

        risk_rows = await conn.fetch(
            """
            SELECT patient_risk_profile, COUNT(*) AS count
            FROM digital_twin_simulations
            WHERE patient_id = $1
            GROUP BY patient_risk_profile
            """,
            patient_id,
        )

        rec_rows = await conn.fetch(
            """
            SELECT recommended_option, COUNT(*) AS count
            FROM digital_twin_simulations
            WHERE patient_id = $1
            GROUP BY recommended_option
            ORDER BY count DESC
            """,
            patient_id,
        )

        avg_row = await conn.fetchrow(
            """
            SELECT
                AVG(baseline_mortality_30d) AS avg_mortality,
                AVG(baseline_readmission_30d) AS avg_readmission,
                AVG(recommendation_confidence) AS avg_confidence
            FROM digital_twin_simulations
            WHERE patient_id = $1
            """,
            patient_id,
        )

        conf_rows = await conn.fetch(
            """
            SELECT model_confidence, COUNT(*) AS count
            FROM digital_twin_simulations
            WHERE patient_id = $1
            GROUP BY model_confidence
            """,
            patient_id,
        )

        most_common_dx = await conn.fetchrow(
            """
            SELECT diagnosis, COUNT(*) AS count
            FROM digital_twin_simulations
            WHERE patient_id = $1
            GROUP BY diagnosis
            ORDER BY count DESC
            LIMIT 1
            """,
            patient_id,
        )

        source_rows = await conn.fetch(
            """
            SELECT source, COUNT(*) AS count
            FROM digital_twin_simulations
            WHERE patient_id = $1
            GROUP BY source
            """,
            patient_id,
        )

        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )

    return PatientSimulationStatsResponse(
        patient_id=patient_id,
        total_simulations=total,
        unique_diagnoses=unique_dx or 0,
        risk_profile_breakdown={r["patient_risk_profile"]: r["count"] for r in risk_rows},
        recommendation_trends={r["recommended_option"]: r["count"] for r in rec_rows},
        average_mortality_risk=round(float(avg_row["avg_mortality"] or 0.0), 4),
        average_readmission_risk=round(float(avg_row["avg_readmission"] or 0.0), 4),
        average_recommendation_confidence=round(float(avg_row["avg_confidence"] or 0.0), 2),
        model_confidence_breakdown={r["model_confidence"]: r["count"] for r in conf_rows},
        most_common_diagnosis=most_common_dx["diagnosis"] if most_common_dx else None,
        most_recommended_option=rec_rows[0]["recommended_option"] if rec_rows else None,
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_simulation=str(latest) if latest else None,
        first_simulation=str(first) if first else None,
    )


@router.get("/compare/{patient_id}", response_model=ComparisonResponse)
async def compare_patient_simulations(patient_id: str):
    """
    Compare simulation outcomes across all sessions for a patient.
    Shows risk trajectories, treatment effectiveness patterns, and outcome trends.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                request_id, created_at, diagnosis, patient_risk_profile,
                baseline_mortality_30d, baseline_readmission_30d,
                recommended_option, recommendation_confidence,
                scenarios
            FROM digital_twin_simulations
            WHERE patient_id = $1
            ORDER BY created_at ASC
            """,
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation records found for patient '{patient_id}'",
        )

    mortality_trajectory = []
    readmission_trajectory = []
    
    for r in rows:
        mortality_trajectory.append({
            "timestamp": str(r["created_at"]),
            "diagnosis": r["diagnosis"],
            "risk": float(r["baseline_mortality_30d"]),
        })
        readmission_trajectory.append({
            "timestamp": str(r["created_at"]),
            "diagnosis": r["diagnosis"],
            "risk": float(r["baseline_readmission_30d"]),
        })

    treatment_outcomes = {}
    for r in rows:
        scenarios = json.loads(r["scenarios"]) if isinstance(r["scenarios"], str) else r["scenarios"]
        for scenario in scenarios:
            opt_id = scenario.get("option_id")
            if opt_id == "C":
                continue
            
            if opt_id not in treatment_outcomes:
                treatment_outcomes[opt_id] = {
                    "label": scenario.get("label", opt_id),
                    "times_recommended": 0,
                    "avg_mortality_reduction": 0.0,
                    "avg_recovery_probability": 0.0,
                    "scenarios_count": 0,
                }
            
            if opt_id == r["recommended_option"]:
                treatment_outcomes[opt_id]["times_recommended"] += 1
            
            preds = scenario.get("predictions", {})
            treatment_outcomes[opt_id]["avg_mortality_reduction"] += (
                float(r["baseline_mortality_30d"]) - preds.get("mortality_risk_30d", 0)
            )
            treatment_outcomes[opt_id]["avg_recovery_probability"] += preds.get("recovery_probability_7d", 0)
            treatment_outcomes[opt_id]["scenarios_count"] += 1

    for opt_id, data in treatment_outcomes.items():
        count = data["scenarios_count"]
        if count > 0:
            data["avg_mortality_reduction"] = round(data["avg_mortality_reduction"] / count, 4)
            data["avg_recovery_probability"] = round(data["avg_recovery_probability"] / count, 4)

    comparison_summary = {
        "total_simulations": len(rows),
        "date_range": {
            "first": str(rows[0]["created_at"]),
            "latest": str(rows[-1]["created_at"]),
        },
        "risk_trends": {
            "mortality_trend": "improving" if mortality_trajectory[-1]["risk"] < mortality_trajectory[0]["risk"] else "worsening",
            "baseline_to_latest_change": round(
                mortality_trajectory[-1]["risk"] - mortality_trajectory[0]["risk"], 4
            ),
        },
    }

    return ComparisonResponse(
        patient_id=patient_id,
        total_simulations=len(rows),
        comparison_summary=comparison_summary,
        risk_trajectories={
            "mortality_30d": mortality_trajectory,
            "readmission_30d": readmission_trajectory,
        },
        treatment_effectiveness=treatment_outcomes,
    )


@router.delete("/{patient_id}")
async def delete_patient_simulation_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """Delete all simulation records for a patient. Requires internal token."""
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM digital_twin_simulations WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"Deleted {deleted} simulation records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}