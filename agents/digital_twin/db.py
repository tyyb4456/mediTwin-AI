"""
agents/digital_twin/db.py
---------------------------
PostgreSQL persistence for digital twin simulation results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /simulate and /stream call:
    await db.save_simulation(record: SimulationRecord)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("meditwin.digital_twin.db")

# ── Pool singleton ─────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create connection pool and ensure table exists."""
    global _pool
    dsn = os.getenv("POSTGRES_CHECKPOINT_URI", "postgresql://postgres:postgres@localhost:5432/meditwin")
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_table()
        logger.info("   ✔   PostgreSQL pool ready (digital_twin)")
    except Exception as e:
        logger.warning(f"  ⚠  PostgreSQL unavailable ({e}) — simulation results will NOT be persisted")
        _pool = None


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def is_available() -> bool:
    return _pool is not None


# ── Table DDL ──────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS digital_twin_simulations (
    id                          SERIAL PRIMARY KEY,
    request_id                  VARCHAR(64) UNIQUE NOT NULL,
    patient_id                  TEXT NOT NULL,
    diagnosis                   TEXT NOT NULL,
    diagnosis_code              TEXT,
    patient_risk_profile        TEXT,
    baseline_mortality_30d      NUMERIC(6,4),
    baseline_readmission_30d    NUMERIC(6,4),
    baseline_complication       NUMERIC(6,4),
    recommended_option          TEXT NOT NULL,
    recommendation_confidence   NUMERIC(4,2),
    model_confidence            TEXT,
    treatment_options_count     INTEGER,
    scenarios                   JSONB NOT NULL,
    simulation_summary          JSONB NOT NULL,
    what_if_narrative           TEXT,
    fhir_care_plan              JSONB,
    feature_attribution         JSONB,
    sensitivity_analysis        JSONB,
    cost_effectiveness          JSONB,
    models_loaded               BOOLEAN DEFAULT TRUE,
    cache_hit                   BOOLEAN DEFAULT FALSE,
    elapsed_ms                  INTEGER,
    source                      TEXT DEFAULT 'simulate',
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_twin_patient_id     ON digital_twin_simulations(patient_id);
CREATE INDEX IF NOT EXISTS idx_twin_created_at     ON digital_twin_simulations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_twin_diagnosis_code ON digital_twin_simulations(diagnosis_code);
CREATE INDEX IF NOT EXISTS idx_twin_risk_profile   ON digital_twin_simulations(patient_risk_profile);
CREATE INDEX IF NOT EXISTS idx_twin_recommended    ON digital_twin_simulations(recommended_option);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class SimulationRecord:
    request_id:                 str
    patient_id:                 str
    diagnosis:                  str
    diagnosis_code:             Optional[str]
    patient_risk_profile:       str
    baseline_mortality_30d:     float
    baseline_readmission_30d:   float
    baseline_complication:      float
    recommended_option:         str
    recommendation_confidence:  float
    model_confidence:           str
    treatment_options_count:    int
    scenarios:                  list
    simulation_summary:         dict
    what_if_narrative:          str
    fhir_care_plan:             Optional[dict]
    feature_attribution:        list
    sensitivity_analysis:       Optional[list]
    cost_effectiveness:         Optional[dict]
    models_loaded:              bool
    cache_hit:                  bool = False
    elapsed_ms:                 Optional[int] = None
    source:                     str = "simulate"


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO digital_twin_simulations (
    request_id, patient_id, diagnosis, diagnosis_code,
    patient_risk_profile,
    baseline_mortality_30d, baseline_readmission_30d, baseline_complication,
    recommended_option, recommendation_confidence, model_confidence,
    treatment_options_count, scenarios, simulation_summary,
    what_if_narrative, fhir_care_plan, feature_attribution,
    sensitivity_analysis, cost_effectiveness,
    models_loaded, cache_hit, elapsed_ms, source
) VALUES (
    $1, $2, $3, $4,
    $5,
    $6, $7, $8,
    $9, $10, $11,
    $12, $13, $14,
    $15, $16, $17,
    $18, $19,
    $20, $21, $22, $23
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_simulation(record: SimulationRecord) -> Optional[int]:
    """
    Persist a digital twin simulation result. Returns inserted row id, or None if skipped/unavailable.
    Never raises — DB failures are logged but don't break the API response.
    """
    if not _pool:
        return None

    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                _INSERT,
                record.request_id,
                record.patient_id,
                record.diagnosis,
                record.diagnosis_code,
                record.patient_risk_profile,
                record.baseline_mortality_30d,
                record.baseline_readmission_30d,
                record.baseline_complication,
                record.recommended_option,
                record.recommendation_confidence,
                record.model_confidence,
                record.treatment_options_count,
                json.dumps(record.scenarios),
                json.dumps(record.simulation_summary),
                record.what_if_narrative,
                json.dumps(record.fhir_care_plan) if record.fhir_care_plan else None,
                json.dumps(record.feature_attribution),
                json.dumps(record.sensitivity_analysis) if record.sensitivity_analysis else None,
                json.dumps(record.cost_effectiveness) if record.cost_effectiveness else None,
                record.models_loaded,
                record.cache_hit,
                record.elapsed_ms,
                record.source,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Digital twin simulation saved → row id={row['id']} "
                f"patient={record.patient_id} recommended={record.recommended_option}"
            )
            return row["id"]
        else:
            logger.debug(f"[{record.request_id}] Duplicate request_id — skipped insert")
            return None

    except Exception as e:
        logger.error(f"[{record.request_id}] DB save failed (non-fatal): {e}")
        return None


# ── Query helpers ──────────────────────────────────────────────────────────────

async def get_by_patient(patient_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent simulation records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM digital_twin_simulations
               WHERE patient_id = $1
               ORDER BY created_at DESC LIMIT $2""",
            patient_id, limit,
        )
    return [dict(r) for r in rows]


async def get_by_request_id(request_id: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM digital_twin_simulations WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None