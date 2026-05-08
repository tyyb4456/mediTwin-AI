"""
agents/lab_analysis/db.py
--------------------------
PostgreSQL persistence for lab analysis results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Mirrors agents/diagnosis/db.py pattern exactly.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /analyze-labs and /stream call:
    await db.save_lab_analysis(record: LabAnalysisRecord)
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

logger = logging.getLogger("meditwin.lab_analysis.db")

# ── Pool singleton ─────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create connection pool and ensure table exists."""
    global _pool
    dsn = os.getenv("POSTGRES_CHECKPOINT_URI", "postgresql://postgres:postgres@localhost:5432/meditwin")
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_table()
        logger.info("  ✔  PostgreSQL pool ready (lab_analysis)")
    except Exception as e:
        logger.warning(f"  ⚠  PostgreSQL unavailable ({e}) — lab results will NOT be persisted")
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
CREATE TABLE IF NOT EXISTS lab_analysis_results (
    id                          SERIAL PRIMARY KEY,
    request_id                  VARCHAR(16) UNIQUE NOT NULL,
    patient_id                  TEXT NOT NULL,
    total_results               INTEGER,
    abnormal_count              INTEGER,
    critical_count              INTEGER,
    overall_severity            TEXT,
    severity_score              JSONB,
    flagged_results             JSONB,
    identified_patterns         JSONB,
    critical_alerts             JSONB,
    confirms_top_diagnosis      BOOLEAN,
    proposed_diagnosis          TEXT,
    proposed_icd10              TEXT,
    lab_confidence_boost        NUMERIC(4,2),
    alternative_diagnosis_code  TEXT,
    clinical_decision_support   JSONB,
    trend_analysis              JSONB,
    llm_interpretation_available BOOLEAN DEFAULT FALSE,
    cache_hit                   BOOLEAN DEFAULT FALSE,
    elapsed_ms                  INTEGER,
    source                      TEXT DEFAULT 'analyze-labs',  -- 'analyze-labs' | 'stream'
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lab_patient_id   ON lab_analysis_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_created_at   ON lab_analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lab_icd10        ON lab_analysis_results(proposed_icd10);
CREATE INDEX IF NOT EXISTS idx_lab_severity     ON lab_analysis_results(overall_severity);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class LabAnalysisRecord:
    request_id:                  str
    patient_id:                  str
    total_results:               int
    abnormal_count:              int
    critical_count:              int
    overall_severity:            str
    severity_score:              Optional[dict]
    flagged_results:             list
    identified_patterns:         list
    critical_alerts:             list
    confirms_top_diagnosis:      bool
    proposed_diagnosis:          str
    proposed_icd10:              str
    lab_confidence_boost:        float
    alternative_diagnosis_code:  Optional[str]
    clinical_decision_support:   Optional[dict]
    trend_analysis:              Optional[list]
    llm_interpretation_available: bool
    cache_hit:                   bool = False
    elapsed_ms:                  Optional[int] = None
    source:                      str = "analyze-labs"   # "analyze-labs" | "stream"


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO lab_analysis_results (
    request_id, patient_id,
    total_results, abnormal_count, critical_count, overall_severity,
    severity_score, flagged_results, identified_patterns, critical_alerts,
    confirms_top_diagnosis, proposed_diagnosis, proposed_icd10,
    lab_confidence_boost, alternative_diagnosis_code,
    clinical_decision_support, trend_analysis,
    llm_interpretation_available, cache_hit, elapsed_ms, source
) VALUES (
    $1, $2,
    $3, $4, $5, $6,
    $7, $8, $9, $10,
    $11, $12, $13,
    $14, $15,
    $16, $17,
    $18, $19, $20, $21
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_lab_analysis(record: LabAnalysisRecord) -> Optional[int]:
    """
    Persist a lab analysis result. Returns inserted row id, or None if skipped/unavailable.
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
                record.total_results,
                record.abnormal_count,
                record.critical_count,
                record.overall_severity,
                json.dumps(record.severity_score) if record.severity_score else None,
                json.dumps(record.flagged_results),
                json.dumps(record.identified_patterns),
                json.dumps(record.critical_alerts),
                record.confirms_top_diagnosis,
                record.proposed_diagnosis,
                record.proposed_icd10,
                record.lab_confidence_boost,
                record.alternative_diagnosis_code,
                json.dumps(record.clinical_decision_support) if record.clinical_decision_support else None,
                json.dumps(record.trend_analysis) if record.trend_analysis else None,
                record.llm_interpretation_available,
                record.cache_hit,
                record.elapsed_ms,
                record.source,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Lab analysis saved → row id={row['id']} "
                f"patient={record.patient_id}"
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
    """Fetch recent lab analysis records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM lab_analysis_results
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
            "SELECT * FROM lab_analysis_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None