"""
agents/imaging_triage/db.py
-----------------------------
PostgreSQL persistence for imaging triage results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Mirrors agents/diagnosis/db.py pattern exactly.

Setup:
    Set env var:  POSTGRES_CHECKPOINT_URI=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /analyze-xray and /stream call:
    await db.save_imaging_result(record: ImagingRecord)
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

logger = logging.getLogger("meditwin.imaging_triage.db")

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
        logger.info("  ✔  PostgreSQL pool ready (imaging_triage)")
    except Exception as e:
        logger.warning(
            f"  ⚠  PostgreSQL unavailable ({e}) — imaging results will NOT be persisted"
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
CREATE TABLE IF NOT EXISTS imaging_triage_results (
    id                      SERIAL PRIMARY KEY,
    request_id              VARCHAR(16) UNIQUE NOT NULL,
    patient_id              TEXT NOT NULL,

    -- Analysis mode
    analysis_mode           TEXT NOT NULL DEFAULT 'cnn',   -- 'cnn' | 'synthetic' | 'mock'
    model_loaded            BOOLEAN DEFAULT FALSE,
    image_provided          BOOLEAN DEFAULT FALSE,

    -- CNN output (real inference)
    prediction              TEXT,                           -- PNEUMONIA | NORMAL | MOCK_NO_MODEL
    confidence              NUMERIC(6, 4),
    pneumonia_probability   NUMERIC(6, 4),
    normal_probability      NUMERIC(6, 4),

    -- Triage
    triage_grade            TEXT,                           -- SEVERE | MODERATE | MILD | NORMAL
    triage_priority         INTEGER,                        -- 1-4
    triage_label            TEXT,                           -- IMMEDIATE | URGENT | SEMI-URGENT | ROUTINE

    -- Imaging findings
    pattern                 TEXT,
    affected_area           TEXT,
    bilateral               BOOLEAN DEFAULT FALSE,
    confidence_in_findings  TEXT,

    -- Clinical
    clinical_interpretation TEXT,
    confirms_diagnosis      BOOLEAN DEFAULT FALSE,
    diagnosis_code          TEXT,                           -- ICD-10 if PNEUMONIA

    -- Patient context used
    patient_age             INTEGER,
    patient_gender          TEXT,
    chief_complaint         TEXT,

    -- LLM synthetic mode
    llm_enriched            BOOLEAN DEFAULT FALSE,
    llm_token_count         INTEGER DEFAULT 0,
    synthetic_reasoning     TEXT,                           -- LLM clinical reasoning narrative

    -- FHIR
    fhir_diagnostic_report  JSONB,

    -- Metadata
    mock                    BOOLEAN DEFAULT FALSE,
    cache_hit               BOOLEAN DEFAULT FALSE,
    elapsed_ms              INTEGER,
    source                  TEXT DEFAULT 'analyze-xray',   -- 'analyze-xray' | 'stream'
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_img_patient_id    ON imaging_triage_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_img_created_at    ON imaging_triage_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_img_prediction    ON imaging_triage_results(prediction);
CREATE INDEX IF NOT EXISTS idx_img_triage_grade  ON imaging_triage_results(triage_grade);
CREATE INDEX IF NOT EXISTS idx_img_mode          ON imaging_triage_results(analysis_mode);
CREATE INDEX IF NOT EXISTS idx_img_source        ON imaging_triage_results(source);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class ImagingRecord:
    request_id:             str
    patient_id:             str
    analysis_mode:          str             # 'cnn' | 'synthetic' | 'mock'
    model_loaded:           bool
    image_provided:         bool

    # Model output
    prediction:             Optional[str]   # PNEUMONIA | NORMAL | MOCK_NO_MODEL
    confidence:             Optional[float]
    pneumonia_probability:  Optional[float]
    normal_probability:     Optional[float]

    # Triage
    triage_grade:           Optional[str]
    triage_priority:        Optional[int]
    triage_label:           Optional[str]

    # Findings
    pattern:                Optional[str]
    affected_area:          Optional[str]
    bilateral:              bool
    confidence_in_findings: Optional[str]

    # Clinical
    clinical_interpretation: Optional[str]
    confirms_diagnosis:      bool
    diagnosis_code:          Optional[str]

    # Patient context
    patient_age:             Optional[int]
    patient_gender:          Optional[str]
    chief_complaint:         Optional[str]

    # LLM
    llm_enriched:            bool = False
    llm_token_count:         int = 0
    synthetic_reasoning:     Optional[str] = None

    # FHIR
    fhir_diagnostic_report:  Optional[dict] = None

    mock:                    bool = False
    cache_hit:               bool = False
    elapsed_ms:              Optional[int] = None
    source:                  str = "analyze-xray"   # "analyze-xray" | "stream"


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO imaging_triage_results (
    request_id, patient_id,
    analysis_mode, model_loaded, image_provided,
    prediction, confidence, pneumonia_probability, normal_probability,
    triage_grade, triage_priority, triage_label,
    pattern, affected_area, bilateral, confidence_in_findings,
    clinical_interpretation, confirms_diagnosis, diagnosis_code,
    patient_age, patient_gender, chief_complaint,
    llm_enriched, llm_token_count, synthetic_reasoning,
    fhir_diagnostic_report,
    mock, cache_hit, elapsed_ms, source
) VALUES (
    $1, $2,
    $3, $4, $5,
    $6, $7, $8, $9,
    $10, $11, $12,
    $13, $14, $15, $16,
    $17, $18, $19,
    $20, $21, $22,
    $23, $24, $25,
    $26,
    $27, $28, $29, $30
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_imaging_result(record: ImagingRecord) -> Optional[int]:
    """
    Persist an imaging triage result.
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
                record.analysis_mode,
                record.model_loaded,
                record.image_provided,
                record.prediction,
                record.confidence,
                record.pneumonia_probability,
                record.normal_probability,
                record.triage_grade,
                record.triage_priority,
                record.triage_label,
                record.pattern,
                record.affected_area,
                record.bilateral,
                record.confidence_in_findings,
                record.clinical_interpretation,
                record.confirms_diagnosis,
                record.diagnosis_code,
                record.patient_age,
                record.patient_gender,
                record.chief_complaint,
                record.llm_enriched,
                record.llm_token_count,
                record.synthetic_reasoning,
                json.dumps(record.fhir_diagnostic_report) if record.fhir_diagnostic_report else None,
                record.mock,
                record.cache_hit,
                record.elapsed_ms,
                record.source,
            )

        if row:
            logger.info(
                f"[{record.request_id}] Imaging result saved → row id={row['id']} "
                f"patient={record.patient_id} mode={record.analysis_mode} "
                f"prediction={record.prediction}"
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
    """Fetch recent imaging triage records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM imaging_triage_results
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
            "SELECT * FROM imaging_triage_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None