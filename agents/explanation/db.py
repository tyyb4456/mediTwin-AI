"""
agents/explanation/db.py
-------------------------
PostgreSQL persistence for explanation results.
Mirrors agents/diagnosis/db.py exactly — same pool pattern, same non-fatal saves.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@host:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /explain and /stream call:
    await db.save_explanation(record: ExplanationRecord)
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

logger = logging.getLogger("meditwin.explanation.db")

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
        logger.info("✓ PostgreSQL pool ready (explanation)")
    except Exception as e:
        logger.warning(
            f"PostgreSQL unavailable ({e}) — explanation results will NOT be persisted"
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
CREATE TABLE IF NOT EXISTS explanation_results (
    id                      SERIAL PRIMARY KEY,
    request_id              VARCHAR(16) UNIQUE NOT NULL,
    patient_id              TEXT NOT NULL,
    chief_complaint         TEXT NOT NULL,

    -- Core outputs
    consensus_status        TEXT,
    final_diagnosis         TEXT,
    aggregate_confidence    FLOAT,
    human_review_required   BOOLEAN DEFAULT FALSE,

    -- Structured outputs stored as JSONB
    soap_note               JSONB,
    patient_output          JSONB,
    risk_attribution        JSONB,
    fhir_bundle_summary     JSONB,   -- NOT the full bundle — just entry_count + resource_types
    risk_flags              JSONB,

    -- Reading level gate stats
    reading_grade_level     FLOAT,
    reading_acceptable      BOOLEAN,
    reading_attempts        INTEGER DEFAULT 1,

    -- Token streaming stats
    soap_tokens             INTEGER DEFAULT 0,
    patient_tokens          INTEGER DEFAULT 0,

    -- Source + perf
    source                  TEXT DEFAULT 'explain',  -- 'explain' | 'stream'
    elapsed_ms              INTEGER,
    cache_hit               BOOLEAN DEFAULT FALSE,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_explanation_patient_id ON explanation_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_explanation_created_at ON explanation_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_explanation_consensus   ON explanation_results(consensus_status);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class ExplanationRecord:
    request_id:             str
    patient_id:             str
    chief_complaint:        str
    consensus_status:       str
    final_diagnosis:        str
    aggregate_confidence:   float
    human_review_required:  bool
    soap_note:              dict
    patient_output:         dict
    risk_attribution:       dict
    fhir_bundle_summary:    dict        # lightweight — not the full FHIR bundle
    risk_flags:             list
    reading_grade_level:    float
    reading_acceptable:     bool
    reading_attempts:       int = 1
    soap_tokens:            int = 0
    patient_tokens:         int = 0
    source:                 str = "explain"
    elapsed_ms:             Optional[int] = None
    cache_hit:              bool = False


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO explanation_results (
    request_id, patient_id, chief_complaint,
    consensus_status, final_diagnosis, aggregate_confidence, human_review_required,
    soap_note, patient_output, risk_attribution, fhir_bundle_summary, risk_flags,
    reading_grade_level, reading_acceptable, reading_attempts,
    soap_tokens, patient_tokens,
    source, elapsed_ms, cache_hit
) VALUES (
    $1, $2, $3,
    $4, $5, $6, $7,
    $8, $9, $10, $11, $12,
    $13, $14, $15,
    $16, $17,
    $18, $19, $20
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_explanation(record: ExplanationRecord) -> Optional[int]:
    """
    Persist an explanation result. Returns inserted row id, or None if skipped/unavailable.
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
                record.chief_complaint,
                record.consensus_status,
                record.final_diagnosis,
                record.aggregate_confidence,
                record.human_review_required,
                json.dumps(record.soap_note),
                json.dumps(record.patient_output),
                json.dumps(record.risk_attribution),
                json.dumps(record.fhir_bundle_summary),
                json.dumps(record.risk_flags),
                record.reading_grade_level,
                record.reading_acceptable,
                record.reading_attempts,
                record.soap_tokens,
                record.patient_tokens,
                record.source,
                record.elapsed_ms,
                record.cache_hit,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Explanation saved → row id={row['id']} patient={record.patient_id}"
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
    """Fetch recent explanation records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM explanation_results
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
            "SELECT * FROM explanation_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None