"""
agents/lab_analysis/stream_endpoint.py
-----------------------------------------
SSE streaming endpoint for the Lab Analysis Agent.

Fixes applied:
  - astream_events(version="v2") replacing the broken chain.astream() approach,
    matching the diagnosis agent pattern exactly.
  - on_chat_model_stream  → yield evt_token (live tokens to client)
  - on_chain_end / RunnableSequence → capture LabInterpretation structured output
  - DB persistence via save_lab_analysis (same as /analyze-labs)
  - response shape matches /analyze-labs exactly
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Optional, List, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

from lab_logic import (
    LabInterpretation,
    TrendAnalysis,
    analyze_trends,
    generate_clinical_decision_support,
    build_llm_prompt,
    build_llm_chain,
)

from db import save_lab_analysis, LabAnalysisRecord

lab_router = APIRouter()


class LabStreamRequest(BaseModel):
    patient_state: dict
    diagnosis_agent_output: Optional[dict] = None
    previous_lab_results: Optional[List[dict]] = None


async def _lab_stream(
    patient_state: dict,
    diagnosis_output: Optional[dict],
    previous_labs: Optional[list],
) -> AsyncIterator[str]:
    node = "lab_analysis"
    timer = Timer()

    # Import the lifespan-managed singletons from main at call time — avoids
    # circular imports while still picking up the live initialized objects.
    from main import llm, llm_ready  # noqa: PLC0415

    patient_id = patient_state.get("patient_id", "unknown")
    request_id = str(uuid.uuid4())[:8]

    if not llm_ready or not llm:
        yield evt_error(
            node,
            "Lab Analysis LLM not initialized — rules-only mode active. "
            "LLM token streaming unavailable.",
            fatal=False,
        )

    # ── Early exit: no lab data ────────────────────────────────────────────────
    labs = patient_state.get("lab_results", [])
    if not labs:
        proposed_dx    = (diagnosis_output or {}).get("top_diagnosis", "Unknown")
        proposed_icd10 = (diagnosis_output or {}).get("top_icd10_code", "UNKNOWN")

        empty_result = {
            "lab_summary": {
                "total_results":  0,
                "abnormal_count": 0,
                "critical_count": 0,
                "overall_severity": "UNKNOWN",
                "status": "NO_DATA",
            },
            "flagged_results": [],
            "pattern_analysis": {
                "identified_patterns":    [],
                "pattern_interpretation": "No laboratory data available",
            },
            "diagnosis_confirmation": {
                "proposed_diagnosis":            proposed_dx,
                "proposed_icd10":                proposed_icd10,
                "confirms_top_diagnosis":        None,
                "lab_confidence_boost":          0,
                "alternative_diagnosis_code":    None,
                "alternative_diagnosis_display": None,
                "reasoning": "Cannot assess diagnosis without laboratory data",
            },
            "critical_alerts": [],
            "trend_analysis":  None,
            "severity_score": {
                "score":                 0,
                "risk_category":         "UNKNOWN",
                "contributors":          [],
                "organ_systems_affected": 0,
            },
            "clinical_decision_support": {
                "immediate_actions": [],
                "urgent_actions": [
                    {
                        "priority":  "URGENT",
                        "action":    "Obtain baseline laboratory tests immediately",
                        "details":   "CBC, CRP, BMP, Lactate",
                        "timeframe": "Immediate",
                    }
                ],
                "routine_actions":           [],
                "monitoring_plan":           [],
                "consultations_recommended": [],
                "follow_up_labs":            [],
            },
            "llm_interpretation_available": False,
            "request_id": request_id,
            "cache_hit":  False,
        }

        yield evt_error(node, "No laboratory data provided — returning empty result", fatal=False)
        yield evt_complete(node, empty_result, elapsed_ms=timer.elapsed_ms())
        return

    # ── Step 1: Rules engine ───────────────────────────────────────────────────
    yield evt_status(node, "Running deterministic rules engine (CRITICAL flags)...",
                     step=1, total=4)

    demographics = patient_state.get("demographics", {})
    age    = demographics.get("age", 40)
    gender = demographics.get("gender", "male")

    from rules_engine import (
        classify_all, detect_patterns, generate_critical_alerts,
        compute_overall_severity, compute_severity_score,
    )

    classified   = await asyncio.to_thread(classify_all, labs, age, gender)
    abnormal     = [r for r in classified if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    critical     = [r for r in classified if r["flag"] == "CRITICAL"]
    severity     = await asyncio.to_thread(compute_overall_severity, classified)
    sev_score    = await asyncio.to_thread(compute_severity_score, classified)

    yield evt_progress(node,
                       f"Rules engine: {len(abnormal)} abnormal, {len(critical)} critical",
                       pct=20)

    # ── Step 2: Pattern detection ──────────────────────────────────────────────
    yield evt_status(node, "Detecting clinical patterns...", step=2, total=4)

    patterns        = await asyncio.to_thread(detect_patterns, classified)
    critical_alerts = await asyncio.to_thread(generate_critical_alerts, classified)

    yield evt_progress(node,
                       f"Patterns: {len(patterns)} detected, {len(critical_alerts)} critical alerts",
                       pct=40)

    # ── Step 3: Trend analysis ─────────────────────────────────────────────────
    trend_analysis = None
    if previous_labs:
        yield evt_status(node, "Performing trend analysis (delta checks)...",
                         step=3, total=4)
        trend_analysis = await asyncio.to_thread(
            analyze_trends, labs, previous_labs, classified
        )
        yield evt_progress(node, f"Trend analysis: {len(trend_analysis)} trends", pct=55)

    # ── Step 4: Structured LLM interpretation with token streaming ─────────────
    yield evt_status(node, "Running structured LLM lab interpretation (streaming)...",
                     step=4, total=4)

    llm_interp: Optional[LabInterpretation] = None
    token_count = 0

    if llm_ready and llm:
        prompt, inputs = build_llm_prompt(classified, patterns, diagnosis_output, critical_alerts)
        structured_llm = build_llm_chain(llm)
        chain = prompt | structured_llm

        # ── astream_events — identical pattern to diagnosis agent ──────────────
        try:
            async for event in chain.astream_events(inputs, version="v2"):
                kind = event["event"]
                name = event["name"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        yield evt_token(node, chunk.content)

                elif kind == "on_chain_end" and name == "RunnableSequence":
                    raw = event["data"].get("output")
                    if isinstance(raw, LabInterpretation):
                        llm_interp = raw
                    elif isinstance(raw, dict):
                        try:
                            llm_interp = LabInterpretation(**raw)
                        except Exception as parse_exc:
                            yield evt_error(node, f"LabInterpretation parse failed: {parse_exc}")

        except Exception as exc:
            yield evt_error(node, f"LLM streaming failed: {exc} — falling back to sync invoke")
            try:
                llm_interp = await asyncio.to_thread(chain.invoke, inputs)
            except Exception as exc2:
                yield evt_error(node, f"LLM completely failed: {exc2} — rules-only result")

        yield evt_progress(node,
                           f"LLM streamed {token_count} tokens — interpretation ready",
                           pct=85)
    else:
        yield evt_progress(node, "LLM unavailable — rules-only result", pct=85)

    # ── Build clinical decision support ────────────────────────────────────────
    cds = generate_clinical_decision_support(classified, patterns, critical_alerts, llm_interp)

    # ── Assemble diagnosis_confirmation (matches /analyze-labs shape exactly) ──
    proposed_dx    = (diagnosis_output or {}).get("top_diagnosis", "Unknown")
    proposed_icd10 = (diagnosis_output or {}).get("top_icd10_code", "UNKNOWN")

    if llm_interp:
        confirms         = llm_interp.confirms_top_diagnosis
        confidence_boost = llm_interp.lab_confidence_boost
        alt_code         = llm_interp.alternative_diagnoses[0].icd10_code if llm_interp.alternative_diagnoses else None
        alt_display      = llm_interp.alternative_diagnoses[0].display if llm_interp.alternative_diagnoses else None
        reasoning        = llm_interp.reasoning
        pattern_interp   = llm_interp.pattern_interpretation
    else:
        bacterial        = any("bacterial" in p["pattern"].lower() for p in patterns)
        confirms         = not (bacterial and proposed_icd10 and not proposed_icd10.startswith("J"))
        confidence_boost = 0.10 if bacterial else 0.0
        alt_code         = None
        alt_display      = None
        reasoning        = "Rules-only mode"
        pattern_interp   = (
            f"Detected: {', '.join(p['pattern'] for p in patterns)}" if patterns else "No patterns"
        )

    flagged_results = [
        {
            "loinc": r["loinc"],
            "display": r["display"],
            "value": r["value"],
            "unit": r["unit"],
            "reference_range": r["reference_range"],
            "flag": r["flag"],
            "clinical_significance": r["clinical_notes"],
        }
        for r in abnormal
    ]

    elapsed = timer.elapsed_ms()

    # ── Persist to DB (non-fatal — mirrors /analyze-labs) ─────────────────────
    await save_lab_analysis(LabAnalysisRecord(
        request_id=request_id,
        patient_id=patient_id,
        total_results=len(classified),
        abnormal_count=len(abnormal),
        critical_count=len(critical),
        overall_severity=severity,
        severity_score=sev_score,
        flagged_results=flagged_results,
        identified_patterns=patterns,
        critical_alerts=critical_alerts,
        confirms_top_diagnosis=confirms,
        proposed_diagnosis=proposed_dx,
        proposed_icd10=proposed_icd10,
        lab_confidence_boost=confidence_boost,
        alternative_diagnosis_code=alt_code,
        clinical_decision_support=cds,
        trend_analysis=[t.model_dump() for t in trend_analysis] if trend_analysis else None,
        llm_interpretation_available=llm_interp is not None,
        cache_hit=False,
        elapsed_ms=elapsed,
        source="stream",
    ))

    result = {
        "lab_summary": {
            "total_results":    len(classified),
            "abnormal_count":   len(abnormal),
            "critical_count":   len(critical),
            "overall_severity": severity,
        },
        "flagged_results": flagged_results,
        "pattern_analysis": {
            "identified_patterns":    patterns,
            "pattern_interpretation": pattern_interp,
        },
        "diagnosis_confirmation": {
            "proposed_diagnosis":            proposed_dx,
            "proposed_icd10":                proposed_icd10,
            "confirms_top_diagnosis":        confirms,
            "lab_confidence_boost":          round(confidence_boost, 2),
            "alternative_diagnosis_code":    alt_code,
            "alternative_diagnosis_display": alt_display,
            "reasoning":                     reasoning,
        },
        "critical_alerts":               critical_alerts,
        "trend_analysis":                [t.model_dump() for t in trend_analysis] if trend_analysis else None,
        "severity_score":                sev_score,       # top-level only, not in lab_summary
        "clinical_decision_support":     cds,
        "llm_interpretation_available":  llm_interp is not None,
        "request_id":                    request_id,
        "cache_hit":                     False,
    }

    yield evt_complete(node, result, elapsed_ms=elapsed)


@lab_router.post("/stream")
async def lab_stream(request: LabStreamRequest):
    async def gen():
        async for chunk in _lab_stream(
            request.patient_state,
            request.diagnosis_agent_output,
            request.previous_lab_results,
        ):
            yield chunk
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)