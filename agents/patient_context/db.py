"""
agents/patient_context/db.py
------------------------------
PostgreSQL persistence for patient context fetch results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Mirrors agents/diagnosis/db.py and agents/lab_analysis/db.py pattern exactly.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /fetch and /stream call:
    await db.save_patient_context(record: PatientContextRecord)
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

logger = logging.getLogger("meditwin.patient_context.db")

# ── Pool singleton ─────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create connection pool and ensure table exists."""
    global _pool
    dsn = os.getenv("POSTGRES_CHECKPOINT_URI", "postgresql://postgres:postgres@localhost:5432/meditwin")
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_table()
        logger.info("✓ PostgreSQL pool ready (patient_context)")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}) — patient context will NOT be persisted")
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
CREATE TABLE IF NOT EXISTS patient_context_results (
    id                      SERIAL PRIMARY KEY,
    request_id              VARCHAR(16) UNIQUE NOT NULL,
    patient_id              TEXT NOT NULL,
    source                  TEXT NOT NULL DEFAULT 'direct',   -- 'SHARP' | 'direct'
    fhir_base_url           TEXT,
    fhir_resources_fetched  INTEGER DEFAULT 0,
    demographics            JSONB,
    active_conditions       JSONB,
    medications             JSONB,
    allergies               JSONB,
    lab_results             JSONB,
    diagnostic_reports      JSONB,
    imaging_available       BOOLEAN DEFAULT FALSE,
    conditions_count        INTEGER DEFAULT 0,
    medications_count       INTEGER DEFAULT 0,
    allergies_count         INTEGER DEFAULT 0,
    lab_results_count       INTEGER DEFAULT 0,
    diagnostic_reports_count INTEGER DEFAULT 0,
    cache_hit               BOOLEAN DEFAULT FALSE,
    fetch_time_ms           INTEGER,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pc_patient_id  ON patient_context_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_pc_created_at  ON patient_context_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pc_source      ON patient_context_results(source);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class PatientContextRecord:
    request_id:               str
    patient_id:               str
    source:                   str                    # "SHARP" | "direct"
    fhir_base_url:            Optional[str]
    fhir_resources_fetched:   int
    demographics:             Optional[dict]
    active_conditions:        list
    medications:              list
    allergies:                list
    lab_results:              list
    diagnostic_reports:       list
    imaging_available:        bool
    cache_hit:                bool = False
    fetch_time_ms:            Optional[int] = None


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO patient_context_results (
    request_id, patient_id, source, fhir_base_url, fhir_resources_fetched,
    demographics, active_conditions, medications, allergies,
    lab_results, diagnostic_reports, imaging_available,
    conditions_count, medications_count, allergies_count,
    lab_results_count, diagnostic_reports_count,
    cache_hit, fetch_time_ms
) VALUES (
    $1, $2, $3, $4, $5,
    $6, $7, $8, $9,
    $10, $11, $12,
    $13, $14, $15,
    $16, $17,
    $18, $19
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_patient_context(record: PatientContextRecord) -> Optional[int]:
    """
    Persist a patient context fetch result.
    Returns inserted row id, or None if skipped/unavailable.
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
                record.source,
                record.fhir_base_url,
                record.fhir_resources_fetched,
                json.dumps(record.demographics) if record.demographics else None,
                json.dumps(record.active_conditions),
                json.dumps(record.medications),
                json.dumps(record.allergies),
                json.dumps(record.lab_results),
                json.dumps(record.diagnostic_reports),
                record.imaging_available,
                len(record.active_conditions),
                len(record.medications),
                len(record.allergies),
                len(record.lab_results),
                len(record.diagnostic_reports),
                record.cache_hit,
                record.fetch_time_ms,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Patient context saved → row id={row['id']} "
                f"patient={record.patient_id} source={record.source}"
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
    """Fetch recent patient context records."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM patient_context_results
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
            "SELECT * FROM patient_context_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None