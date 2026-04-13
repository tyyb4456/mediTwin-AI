"""
agents/drug_safety/db.py
--------------------------
PostgreSQL persistence for drug safety check results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Mirrors agents/diagnosis/db.py and agents/lab_analysis/db.py pattern exactly.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /check-safety and /stream call:
    await db.save_drug_safety(record: DrugSafetyRecord)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("meditwin.drug_safety.db")

# ── Pool singleton ─────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create connection pool and ensure table exists."""
    global _pool
    dsn = os.getenv(
        "POSTGRES_CHECKPOINT_URI",
        "postgresql://postgres:postgres@localhost:5432/meditwin",
    )
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_table()
        logger.info("✓ PostgreSQL pool ready (drug_safety)")
    except Exception as e:
        logger.warning(
            f"PostgreSQL unavailable ({e}) — drug safety results will NOT be persisted"
        )
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
CREATE TABLE IF NOT EXISTS drug_safety_results (
    id                          SERIAL PRIMARY KEY,
    request_id                  VARCHAR(16) UNIQUE NOT NULL,
    patient_id                  TEXT NOT NULL,

    -- Input summary
    proposed_medications        JSONB,          -- list[str]
    current_medications         JSONB,          -- list[str]
    proposed_count              INTEGER DEFAULT 0,
    current_count               INTEGER DEFAULT 0,

    -- Core safety verdict
    safety_status               TEXT NOT NULL,  -- SAFE | CAUTION | UNSAFE

    -- Deterministic findings
    contraindications           JSONB,          -- list[dict]
    contraindication_count      INTEGER DEFAULT 0,
    approved_medications        JSONB,          -- list[str]
    flagged_medications         JSONB,          -- list[str]
    approved_count              INTEGER DEFAULT 0,
    flagged_count               INTEGER DEFAULT 0,

    -- External API results
    critical_interactions       JSONB,          -- list[dict] (enriched)
    interaction_count           INTEGER DEFAULT 0,
    fda_warnings                JSONB,          -- dict[drug -> list[str]]
    black_box_count             INTEGER DEFAULT 0,

    -- LLM enrichment
    patient_risk_profile        JSONB,          -- PatientRiskProfile.model_dump()
    interaction_risk_narrative  TEXT,
    overall_risk_level          TEXT,           -- extracted from patient_risk_profile
    recommended_action          TEXT,           -- extracted from patient_risk_profile
    llm_enriched                BOOLEAN DEFAULT FALSE,

    -- FHIR output
    fhir_medication_requests    JSONB,          -- list[dict]

    -- Metadata
    cache_hit                   BOOLEAN DEFAULT FALSE,
    elapsed_ms                  INTEGER,
    source                      TEXT DEFAULT 'check-safety',  -- 'check-safety' | 'stream'
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ds_patient_id      ON drug_safety_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_ds_created_at      ON drug_safety_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ds_safety_status   ON drug_safety_results(safety_status);
CREATE INDEX IF NOT EXISTS idx_ds_risk_level      ON drug_safety_results(overall_risk_level);
CREATE INDEX IF NOT EXISTS idx_ds_source          ON drug_safety_results(source);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class DrugSafetyRecord:
    request_id:                 str
    patient_id:                 str

    proposed_medications:       list
    current_medications:        list

    safety_status:              str             # SAFE | CAUTION | UNSAFE

    contraindications:          list
    approved_medications:       list
    flagged_medications:        list

    critical_interactions:      list            # enriched RxNav interactions
    fda_warnings:               dict            # {drug: [warning_str]}

    patient_risk_profile:       Optional[dict]
    interaction_risk_narrative: Optional[str]
    fhir_medication_requests:   list

    llm_enriched:               bool = False
    cache_hit:                  bool = False
    elapsed_ms:                 Optional[int] = None
    source:                     str = "check-safety"   # "check-safety" | "stream"


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO drug_safety_results (
    request_id, patient_id,
    proposed_medications, current_medications,
    proposed_count, current_count,
    safety_status,
    contraindications, contraindication_count,
    approved_medications, flagged_medications,
    approved_count, flagged_count,
    critical_interactions, interaction_count,
    fda_warnings, black_box_count,
    patient_risk_profile, interaction_risk_narrative,
    overall_risk_level, recommended_action,
    llm_enriched,
    fhir_medication_requests,
    cache_hit, elapsed_ms, source
) VALUES (
    $1, $2,
    $3, $4,
    $5, $6,
    $7,
    $8, $9,
    $10, $11,
    $12, $13,
    $14, $15,
    $16, $17,
    $18, $19,
    $20, $21,
    $22,
    $23,
    $24, $25, $26
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_drug_safety(record: DrugSafetyRecord) -> Optional[int]:
    """
    Persist a drug safety check result.
    Returns inserted row id, or None if skipped/unavailable.
    Never raises — DB failures are logged but don't break the API response.
    """
    if not _pool:
        return None

    try:
        # Extract top-level fields from nested risk profile
        overall_risk_level = None
        recommended_action = None
        if record.patient_risk_profile:
            overall_risk_level = record.patient_risk_profile.get("overall_risk_level")
            recommended_action = record.patient_risk_profile.get("recommended_action")

        # Count black box warnings
        black_box_count = sum(
            1 for warnings in record.fda_warnings.values()
            if any("[BLACK BOX]" in w for w in warnings)
        )

        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                _INSERT,
                record.request_id,
                record.patient_id,
                json.dumps(record.proposed_medications),
                json.dumps(record.current_medications),
                len(record.proposed_medications),
                len(record.current_medications),
                record.safety_status,
                json.dumps(record.contraindications),
                len(record.contraindications),
                json.dumps(record.approved_medications),
                json.dumps(record.flagged_medications),
                len(record.approved_medications),
                len(record.flagged_medications),
                json.dumps(record.critical_interactions),
                len(record.critical_interactions),
                json.dumps(record.fda_warnings),
                black_box_count,
                json.dumps(record.patient_risk_profile) if record.patient_risk_profile else None,
                record.interaction_risk_narrative,
                overall_risk_level,
                recommended_action,
                record.llm_enriched,
                json.dumps(record.fhir_medication_requests),
                record.cache_hit,
                record.elapsed_ms,
                record.source,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Drug safety saved → row id={row['id']} "
                f"patient={record.patient_id} status={record.safety_status}"
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
    """Fetch recent drug safety records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM drug_safety_results
               WHERE patient_id = $1
               ORDER BY created_at DESC LIMIT $2""",
            patient_id,
            limit,
        )
    return [dict(r) for r in rows]


async def get_by_request_id(request_id: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM drug_safety_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None