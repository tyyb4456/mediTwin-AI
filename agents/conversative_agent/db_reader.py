"""
agents/conversative_agent/db_reader.py
---------------------------------------
Shared asyncpg connection pool + direct DB query functions for all
MediTwin result tables.

Instead of calling each agent over HTTP, the tool agent reads the most
recent stored result for a given patient_id directly from PostgreSQL.

Tables queried:
    patient_context_results     → fetch_patient_context tool
    diagnosis_results           → run_diagnosis tool
    lab_analysis_results        → analyze_labs tool
    drug_safety_results         → check_drug_safety tool
    imaging_triage_results      → analyze_chest_xray tool
    digital_twin_simulations    → simulate_treatment_outcomes tool

Lifecycle:
    Call db_reader.init() on startup, db_reader.close() on shutdown.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("meditwin.tool_agent.db_reader")

# ── Shared pool singleton ──────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create the shared connection pool. Called once at agent startup."""
    global _pool
    dsn = (
        os.getenv("POSTGRES_CHECKPOINT_URI")
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/meditwin_checkpoints"
    )
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        logger.info("   ✔   DB reader pool ready (tool_agent)")
    except Exception as e:
        logger.warning(f"   ⚠   DB reader unavailable ({e}) — tools will report no cached data")
        _pool = None


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def is_available() -> bool:
    return _pool is not None

# ── JSON deserializer helper ───────────────────────────────────────────────────

def _load(val):
    """asyncpg returns JSONB as str in some versions, dict/list in others."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val
    

# ── Patient Context ────────────────────────────────────────────────────────────

async def get_latest_patient_context(patient_id: str) -> Optional[dict]:
    """
    Return the most recent patient_context_results row for patient_id,
    reconstructed as the patient_state dict expected by downstream tools.
    Returns None if no record exists or DB is unavailable.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM patient_context_results
                WHERE patient_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                patient_id,
            )
        if not row:
            return None

        row = dict(row)
        return {
            "patient_id":          row["patient_id"],
            "source":              row.get("source", "db"),
            "fhir_base_url":       row.get("fhir_base_url"),
            "demographics":        _load(row.get("demographics")) or {},
            "active_conditions":   _load(row.get("active_conditions")) or [],
            "medications":         _load(row.get("medications")) or [],
            "allergies":           _load(row.get("allergies")) or [],
            "lab_results":         _load(row.get("lab_results")) or [],
            "diagnostic_reports":  _load(row.get("diagnostic_reports")) or [],
            "imaging_available":   row.get("imaging_available", False),
            "cache_hit":           True,
            "_from_db":            True,
            "_fetched_at":         str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [patient_context] DB query failed: {e}")
        return None

# ── Diagnosis ──────────────────────────────────────────────────────────────────

async def get_latest_diagnosis(patient_id: str) -> Optional[dict]:
    """
    Return the most recent diagnosis_results row for patient_id.
    Reconstructed to match the shape returned by the diagnosis agent's /diagnose endpoint.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM diagnosis_results
                WHERE patient_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                patient_id,
            )
        if not row:
            return None

        row = dict(row)
        flags = _load(row.get("flags")) or {}
        return {
            "patient_id":              row["patient_id"],
            "chief_complaint":         row.get("chief_complaint", ""),
            "top_diagnosis":           row.get("top_diagnosis", ""),
            "top_icd10_code":          row.get("top_icd10_code", ""),
            "confidence_level":        row.get("confidence_level", ""),
            "rag_mode":                row.get("rag_mode", ""),
            "differential_diagnosis":  _load(row.get("differential_diagnosis")) or [],
            "recommended_next_steps":  _load(row.get("recommended_next_steps")) or [],
            "fhir_conditions":         _load(row.get("fhir_conditions")) or [],
            "penicillin_allergy_flagged": flags.get("penicillin_alert", False),
            "high_suspicion_sepsis":      flags.get("sepsis_alert", False),
            "requires_isolation":         flags.get("requires_isolation", False),
            "cache_hit":               True,
            "_from_db":                True,
            "_fetched_at":             str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [diagnosis] DB query failed: {e}")
        return None

# ── Lab Analysis ───────────────────────────────────────────────────────────────

async def get_latest_lab_analysis(patient_id: str) -> Optional[dict]:
    """
    Return the most recent lab_analysis_results row for patient_id,
    shaped to match the lab analysis agent's /analyze-labs response.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
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
            return None

        row = dict(row)
        return {
            "patient_id":    row["patient_id"],
            "lab_summary": {
                "total_results":    row.get("total_results", 0),
                "abnormal_count":   row.get("abnormal_count", 0),
                "critical_count":   row.get("critical_count", 0),
                "overall_severity": row.get("overall_severity", "NORMAL"),
                "severity_score":   _load(row.get("severity_score")),
            },
            "flagged_results":             _load(row.get("flagged_results")) or [],
            "identified_patterns":         _load(row.get("identified_patterns")) or [],
            "critical_alerts":             _load(row.get("critical_alerts")) or [],
            "confirms_top_diagnosis":      row.get("confirms_top_diagnosis", False),
            "proposed_diagnosis":          row.get("proposed_diagnosis", ""),
            "proposed_icd10":              row.get("proposed_icd10", ""),
            "lab_confidence_boost":        float(row.get("lab_confidence_boost") or 0),
            "alternative_diagnosis_code":  row.get("alternative_diagnosis_code"),
            "clinical_decision_support":   _load(row.get("clinical_decision_support")),
            "trend_analysis":              _load(row.get("trend_analysis")) or [],
            "llm_interpretation_available": row.get("llm_interpretation_available", False),
            "cache_hit":   True,
            "_from_db":    True,
            "_fetched_at": str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [lab_analysis] DB query failed: {e}")
        return None

# ── Drug Safety ────────────────────────────────────────────────────────────────

async def get_latest_drug_safety(patient_id: str) -> Optional[dict]:
    """
    Return the most recent drug_safety_results row for patient_id.
    Shaped to match the drug safety agent's /check-safety response.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
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
            return None

        row = dict(row)
        return {
            "patient_id":               row["patient_id"],
            "safety_status":            row.get("safety_status", "UNKNOWN"),
            "proposed_medications":     _load(row.get("proposed_medications")) or [],
            "current_medications":      _load(row.get("current_medications")) or [],
            "contraindications":        _load(row.get("contraindications")) or [],
            "approved_medications":     _load(row.get("approved_medications")) or [],
            "flagged_medications":      _load(row.get("flagged_medications")) or [],
            "critical_interactions":    _load(row.get("critical_interactions")) or [],
            "fda_warnings":             _load(row.get("fda_warnings")) or {},
            "patient_risk_profile":     _load(row.get("patient_risk_profile")),
            "interaction_risk_narrative": row.get("interaction_risk_narrative"),
            "overall_risk_level":       row.get("overall_risk_level"),
            "recommended_action":       row.get("recommended_action"),
            "fhir_medication_requests": _load(row.get("fhir_medication_requests")) or [],
            "llm_enriched":             row.get("llm_enriched", False),
            "cache_hit":   True,
            "_from_db":    True,
            "_fetched_at": str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [drug_safety] DB query failed: {e}")
        return None

# ── Imaging Triage ─────────────────────────────────────────────────────────────

async def get_latest_imaging(patient_id: str) -> Optional[dict]:
    """
    Return the most recent imaging_triage_results row for patient_id,
    shaped to match the imaging triage agent's /analyze-xray response.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
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
            return None

        row = dict(row)
        return {
            "patient_id":  row["patient_id"],
            "model_output": {
                "prediction":            row.get("prediction"),
                "confidence":            float(row.get("confidence") or 0),
                "pneumonia_probability": float(row.get("pneumonia_probability") or 0),
                "normal_probability":    float(row.get("normal_probability") or 0),
            },
            "severity_assessment": {
                "triage_grade":    row.get("triage_grade"),
                "triage_priority": row.get("triage_priority"),
                "triage_label":    row.get("triage_label"),
            },
            "imaging_findings": {
                "pattern":                row.get("pattern"),
                "affected_area":          row.get("affected_area"),
                "bilateral":              row.get("bilateral", False),
                "confidence_in_findings": row.get("confidence_in_findings"),
            },
            "clinical_interpretation": row.get("clinical_interpretation"),
            "confirms_diagnosis":      row.get("confirms_diagnosis", False),
            "diagnosis_code":          row.get("diagnosis_code"),
            "analysis_mode":           row.get("analysis_mode", "cnn"),
            "llm_enriched":            row.get("llm_enriched", False),
            "mock":                    row.get("mock", False),
            "fhir_diagnostic_report":  _load(row.get("fhir_diagnostic_report")),
            "cache_hit":   True,
            "_from_db":    True,
            "_fetched_at": str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [imaging_triage] DB query failed: {e}")
        return None



# ── Digital Twin ───────────────────────────────────────────────────────────────

# ── Conversations ──────────────────────────────────────────────────────────────

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    messages    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);
"""


async def ensure_conversations_table() -> None:
    """Create the conversations table if it doesn't exist."""
    if not _pool:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(_CREATE_CONVERSATIONS)
    except Exception as e:
        logger.error(f"  ✘    [conversations] Table setup failed: {e}")


async def list_conversations() -> list[dict]:
    """Return all conversations ordered by most recently updated."""
    if not _pool:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, session_id, messages, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            )
        return [
            {
                "id":         r["id"],
                "session_id": r["session_id"],
                "messages":   _load(r["messages"]) or [],
                "created_at": r["created_at"].isoformat(),
                "updated_at": r["updated_at"].isoformat(),
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"  ✘    [conversations] list failed: {e}")
        return []


async def create_conversation(conv_id: str, session_id: str) -> bool:
    """Insert a new empty conversation. Returns True on success."""
    if not _pool:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (id, session_id, messages) "
                "VALUES ($1, $2, '[]') ON CONFLICT (id) DO NOTHING",
                conv_id, session_id,
            )
        return True
    except Exception as e:
        logger.error(f"  ✘    [conversations] create failed: {e}")
        return False


async def update_conversation_messages(conv_id: str, messages: list) -> bool:
    """Overwrite the messages array and bump updated_at."""
    if not _pool:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET messages=$1, updated_at=NOW() WHERE id=$2",
                json.dumps(messages), conv_id,
            )
        return True
    except Exception as e:
        logger.error(f"  ✘    [conversations] update failed: {e}")
        return False


async def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation by id."""
    if not _pool:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute("DELETE FROM conversations WHERE id=$1", conv_id)
        return True
    except Exception as e:
        logger.error(f"  ✘    [conversations] delete failed: {e}")
        return False


async def get_latest_simulation(patient_id: str) -> Optional[dict]:
    """
    Return the most recent digital_twin_simulations row for patient_id,
    shaped to match the digital twin agent's /simulate response.
    """
    if not _pool:
        return None
    try:
        async with _pool.acquire() as conn:
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
            return None

        row = dict(row)
        return {
            "patient_id":          row["patient_id"],
            "diagnosis":           row.get("diagnosis", ""),
            "diagnosis_code":      row.get("diagnosis_code"),
            "patient_risk_profile": row.get("patient_risk_profile"),
            "baseline_risks": {
                "mortality_30d":    float(row.get("baseline_mortality_30d") or 0),
                "readmission_30d":  float(row.get("baseline_readmission_30d") or 0),
                "complication":     float(row.get("baseline_complication") or 0),
            },
            "recommended_option":         row.get("recommended_option"),
            "recommendation_confidence":  float(row.get("recommendation_confidence") or 0),
            "model_confidence":           row.get("model_confidence"),
            "treatment_options_count":    row.get("treatment_options_count", 0),
            "scenarios":                  _load(row.get("scenarios")) or [],
            "simulation_summary":         _load(row.get("simulation_summary")) or {},
            "what_if_narrative":          row.get("what_if_narrative"),
            "fhir_care_plan":             _load(row.get("fhir_care_plan")),
            "feature_attribution":        _load(row.get("feature_attribution")) or [],
            "sensitivity_analysis":       _load(row.get("sensitivity_analysis")),
            "cost_effectiveness":         _load(row.get("cost_effectiveness")),
            "models_loaded":              row.get("models_loaded", True),
            "cache_hit":   True,
            "_from_db":    True,
            "_fetched_at": str(row.get("created_at", "")),
        }
    except Exception as e:
        logger.error(f"  ✘    [digital_twin] DB query failed: {e}")
        return None
