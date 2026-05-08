"""
orchestrator/streaming_graph.py
---------------------------------
The streaming orchestration engine.

This is the orchestrator-side counterpart to all the /stream endpoints on the
individual agent microservices.  It replaces the LangGraph StateGraph for the
streaming path — the graph node order is preserved but implemented as a
sequential (with selective parallelism) async pipeline that proxies each
agent's SSE stream back to the client in real time.

Flow mirrors graph.py exactly:
  1. patient_context          → sequential (fatal if fails)
  2. parallel_diagnosis_lab   → concurrent (asyncio.gather)
  3. imaging_triage           → conditional (only if image_data present)
  4. drug_safety              → sequential
  5. digital_twin             → sequential
  6. consensus                → sequential
  7. explanation              → sequential
  final event → complete JSON

Client sees a continuous SSE stream with events from every agent as they run.

───────────────────────────────────────────────────────────────────────────────
Event types the client receives:

  {"type": "status",   "node": "patient_context", "message": "...", "step": 1, "total": 4}
  {"type": "progress", "node": "diagnosis",        "message": "...", "pct": 45.0}
  {"type": "token",    "node": "diagnosis",        "token": "Comm"}
  {"type": "result",   "node": "lab_analysis",     "data": {...}}
  {"type": "complete", "node": "drug_safety",      "data": {...}, "elapsed_ms": 3240}
  {"type": "error",    "node": "imaging_triage",   "message": "...", "fatal": false}
  {"type": "final",    "data": { ...full MediTwin response... }}
  data: [DONE]
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import httpx

from shared.sse_utils import (
    evt_status, evt_error, evt_final, sse_done,
    collect_agent_stream,
    SSE_HEADERS,
)

# ── Service URLs (from env, with defaults) ─────────────────────────────────────

PATIENT_CONTEXT_URL = os.getenv("PATIENT_CONTEXT_URL", "http://localhost:8001")
DIAGNOSIS_URL       = os.getenv("DIAGNOSIS_URL",        "http://localhost:8002")
LAB_ANALYSIS_URL    = os.getenv("LAB_ANALYSIS_URL",     "http://localhost:8003")
DRUG_SAFETY_URL     = os.getenv("DRUG_SAFETY_URL",      "http://localhost:8004")
IMAGING_TRIAGE_URL  = os.getenv("IMAGING_TRIAGE_URL",   "http://localhost:8005")
DIGITAL_TWIN_URL    = os.getenv("DIGITAL_TWIN_URL",     "http://localhost:8006")
CONSENSUS_URL       = os.getenv("CONSENSUS_URL",        "http://localhost:8007")
EXPLANATION_URL     = os.getenv("EXPLANATION_URL",      "http://localhost:8009")


# ── Helper: extract result from collected SSE lines ───────────────────────────

def _extract_result(lines: list[str]) -> Optional[dict]:
    """
    Scan collected SSE lines for the last "complete" or "result" event and
    return its "data" field.
    """
    last: Optional[dict] = None
    for line in lines:
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
            if evt.get("type") in ("complete", "result") and "data" in evt:
                last = evt["data"]
        except (json.JSONDecodeError, KeyError):
            pass
    return last


def _is_fatal(lines: list[str]) -> bool:
    """Return True if any error event in the lines has fatal=True."""
    for line in lines:
        if not line.startswith("data: "):
            continue
        try:
            evt = json.loads(line[6:].strip())
            if evt.get("type") == "error" and evt.get("fatal"):
                return True
        except Exception:
            pass
    return False


# ── Drug safety payload builder (mirrors agent_callers.py logic) ──────────────

def _build_drug_safety_payload(patient_state: dict, diagnosis_output: Optional[dict]) -> dict:
    current_meds = [m.get("drug", "") for m in patient_state.get("medications", [])]
    allergies    = patient_state.get("allergies", [])
    conditions   = patient_state.get("active_conditions", [])

    proposed = []
    if diagnosis_output:
        top_code = diagnosis_output.get("top_icd10_code", "")
        if top_code.startswith("J1"):
            proposed = ["Amoxicillin 1g", "Azithromycin 500mg"]
        elif top_code.startswith("J4"):
            proposed = ["Azithromycin 500mg", "Prednisone 40mg"]
    if not proposed:
        proposed = ["Azithromycin 500mg"]

    return {
        "proposed_medications": proposed,
        "current_medications":  current_meds,
        "patient_allergies":    allergies,
        "active_conditions":    conditions,
        "patient_id":           patient_state.get("patient_id", "unknown"),
    }


def _build_digital_twin_payload(
    patient_state: dict,
    diagnosis_output: Optional[dict],
    drug_safety_output: Optional[dict],
) -> dict:
    approved = (drug_safety_output or {}).get("approved_medications", [])

    treatment_options = []
    if approved:
        treatment_options.append({
            "option_id": "A",
            "label":     f"{approved[0]} — outpatient",
            "drugs":     [approved[0]],
            "interventions": ["O2 supplementation"],
        })
        if len(approved) >= 1:
            treatment_options.append({
                "option_id": "B",
                "label":     "IV therapy + hospitalization",
                "drugs":     approved[:2],
                "interventions": ["Hospitalization", "IV fluids", "Continuous monitoring"],
            })
    else:
        treatment_options = [
            {"option_id": "A", "label": "Azithromycin outpatient",
             "drugs": ["Azithromycin 500mg"], "interventions": ["O2 supplementation"]},
            {"option_id": "B", "label": "IV antibiotics + hospitalization",
             "drugs": ["Ceftriaxone 1g IV", "Azithromycin 500mg"],
             "interventions": ["Hospitalization", "IV fluids"]},
        ]

    dx_str = (
        f"{diagnosis_output.get('top_diagnosis', 'Unknown')} "
        f"({diagnosis_output.get('top_icd10_code', '')})"
        if diagnosis_output else "Unknown"
    )

    return {
        "patient_state":     patient_state,
        "diagnosis":         dx_str,
        "treatment_options": treatment_options,
    }


# ── Main streaming pipeline ────────────────────────────────────────────────────

async def stream_full_analysis(
    patient_id:     str,
    chief_complaint: str,
    fhir_base_url:  str,
    sharp_token:    str,
    image_data:     Optional[str],
) -> AsyncIterator[str]:
    """
    Full MediTwin analysis as a continuous SSE stream.

    Yields SSE strings.  The caller (FastAPI endpoint) wraps this in a
    StreamingResponse.

    State dict tracks agent outputs as they arrive:
      patient_state, diagnosis_output, lab_output,
      drug_safety_output, imaging_output,
      digital_twin_output, consensus_output, final_output
    """

    start_time = time.perf_counter()

    # Shared mutable state — populated as agents complete
    state: dict = {
        "patient_state":      None,
        "diagnosis_output":   None,
        "lab_output":         None,
        "drug_safety_output": None,
        "imaging_output":     None,
        "digital_twin_output": None,
        "consensus_output":   None,
        "final_output":       None,
        "error_log":          [],
        "human_review_required": False,
    }

    # One persistent httpx client for the full pipeline
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:

        # ── NODE 1: Patient Context ────────────────────────────────────────────

        yield evt_status("orchestrator",
                         f"Starting MediTwin analysis for patient {patient_id}",
                         step=0, total=7)

        pc_lines, pc_result = await collect_agent_stream(
            client,
            f"{PATIENT_CONTEXT_URL}/stream",
            {
                "patient_id":    patient_id,
                "fhir_base_url": fhir_base_url,
                "sharp_token":   sharp_token or None,
            },
            node_name="patient_context",
            timeout=30.0,
        )

        # Proxy all patient_context events to client
        for line in pc_lines:
            yield line

        if pc_result is None or _is_fatal(pc_lines):
            yield evt_error("orchestrator",
                            "Patient Context Agent failed — cannot proceed with analysis.",
                            fatal=True)
            yield evt_final({
                "error": "Patient Context Agent failed",
                "patient_id": patient_id,
                "error_log": state["error_log"],
                "elapsed_seconds": round(time.perf_counter() - start_time, 2),
            })
            yield sse_done()
            return

        # pc_result is the data dict from the "complete" event
        state["patient_state"] = pc_result.get("patient_state")

        if state["patient_state"] is None:
            yield evt_error("orchestrator", "Patient state is None after fetch", fatal=True)
            yield sse_done()
            return

        patient_state = state["patient_state"]

        # ── NODE 2: Parallel — Diagnosis + Lab ────────────────────────────────

        yield evt_status("orchestrator",
                         "Running Diagnosis Agent + Lab Analysis in parallel...",
                         step=2, total=7)

        dx_task  = collect_agent_stream(
            client,
            f"{DIAGNOSIS_URL}/stream",
            {
                "patient_state":          patient_state,
                "chief_complaint":        chief_complaint,
                "include_fhir_resources": True,
            },
            node_name="diagnosis",
            timeout=90.0,  # LLM can be slow
        )
        lab_task = collect_agent_stream(
            client,
            f"{LAB_ANALYSIS_URL}/stream",
            {
                "patient_state":         patient_state,
                "diagnosis_agent_output": None,
            },
            node_name="lab_analysis",
            timeout=60.0,
        )

        (dx_lines, dx_result), (lab_lines, lab_result) = await asyncio.gather(
            dx_task, lab_task
        )

        # Proxy both streams (diagnosis first, then lab)
        for line in dx_lines:
            yield line
        for line in lab_lines:
            yield line

        if dx_result is None:
            state["error_log"].append("WARNING: Diagnosis Agent failed")
        else:
            state["diagnosis_output"] = dx_result

        if lab_result is None:
            state["error_log"].append("WARNING: Lab Analysis Agent failed")
        else:
            state["lab_output"] = lab_result

        # ── NODE 3: Imaging Triage (conditional) ──────────────────────────────

        if image_data:
            yield evt_status("orchestrator",
                             "Image detected — running CNN chest X-ray analysis...",
                             step=3, total=7)

            img_lines, img_result = await collect_agent_stream(
                client,
                f"{IMAGING_TRIAGE_URL}/stream",
                {
                    "patient_id": patient_state.get("patient_id", "unknown"),
                    "image_data": {
                        "format":       "base64",
                        "content_type": "image/jpeg",
                        "data":         image_data,
                    },
                    "patient_context": {
                        "age":               patient_state.get("demographics", {}).get("age", 40),
                        "gender":            patient_state.get("demographics", {}).get("gender", "unknown"),
                        "chief_complaint":   patient_state.get("chief_complaint", ""),
                        "current_diagnosis": None,
                    },
                    "patient_state": patient_state,
                },
                node_name="imaging_triage",
                timeout=60.0,
            )
            for line in img_lines:
                yield line

            if img_result is None:
                state["error_log"].append("WARNING: Imaging Triage Agent failed")
            else:
                state["imaging_output"] = img_result
        else:
            yield evt_status("orchestrator",
                             "No image provided — skipping imaging triage",
                             step=3, total=7)

        # ── NODE 4: Drug Safety ────────────────────────────────────────────────

        yield evt_status("orchestrator", "Running Drug Safety check...",
                         step=4, total=7)

        drug_payload = _build_drug_safety_payload(
            patient_state, state.get("diagnosis_output")
        )

        drug_lines, drug_result = await collect_agent_stream(
            client,
            f"{DRUG_SAFETY_URL}/stream",
            drug_payload,
            node_name="drug_safety",
            timeout=45.0,
        )
        for line in drug_lines:
            yield line

        if drug_result is None:
            state["error_log"].append("WARNING: Drug Safety Agent failed")
        else:
            state["drug_safety_output"] = drug_result

        # ── NODE 5: Digital Twin ───────────────────────────────────────────────

        yield evt_status("orchestrator",
                         "Running Digital Twin outcome simulation...",
                         step=5, total=7)

        twin_payload = _build_digital_twin_payload(
            patient_state,
            state.get("diagnosis_output"),
            state.get("drug_safety_output"),
        )

        twin_lines, twin_result = await collect_agent_stream(
            client,
            f"{DIGITAL_TWIN_URL}/stream",
            twin_payload,
            node_name="digital_twin",
            timeout=60.0,
        )
        for line in twin_lines:
            yield line

        if twin_result is None:
            state["error_log"].append("WARNING: Digital Twin Agent failed")
        else:
            state["digital_twin_output"] = twin_result

        # ── NODE 6: Consensus ──────────────────────────────────────────────────

        yield evt_status("orchestrator",
                         "Running Consensus + Escalation Agent...",
                         step=6, total=7)

        consensus_lines, consensus_result = await collect_agent_stream(
            client,
            f"{CONSENSUS_URL}/stream",
            {
                "diagnosis_output":   state.get("diagnosis_output"),
                "lab_output":         state.get("lab_output"),
                "imaging_output":     state.get("imaging_output"),
                "drug_safety_output": state.get("drug_safety_output"),
                "patient_state":      patient_state,
            },
            node_name="consensus",
            timeout=45.0,
        )
        for line in consensus_lines:
            yield line

        if consensus_result is None:
            # Fallback consensus
            dx = state.get("diagnosis_output") or {}
            consensus_result = {
                "consensus_status":       "FULL_CONSENSUS",
                "final_diagnosis":        dx.get("top_diagnosis", "Unknown"),
                "aggregate_confidence":   0.60,
                "human_review_required":  False,
                "conflict_count":         0,
                "conflicts":              [],
                "consensus_summary":      "Consensus agent unavailable — using diagnosis output.",
                "partial_outputs_available": True,
            }
            state["error_log"].append("WARNING: Consensus Agent failed — using fallback")

        state["consensus_output"]    = consensus_result
        state["human_review_required"] = consensus_result.get("human_review_required", False)

        # ── NODE 7: Explanation ────────────────────────────────────────────────

        yield evt_status("orchestrator",
                         "Generating SOAP note, patient explanation, and FHIR bundle...",
                         step=7, total=7)

        exp_lines, exp_result = await collect_agent_stream(
            client,
            f"{EXPLANATION_URL}/stream",
            {
                "patient_state":      patient_state,
                "consensus_output":   consensus_result,
                "diagnosis_output":   state.get("diagnosis_output"),
                "lab_output":         state.get("lab_output"),
                "imaging_output":     state.get("imaging_output"),
                "drug_safety_output": state.get("drug_safety_output"),
                "digital_twin_output": state.get("digital_twin_output"),
                "chief_complaint":    chief_complaint,
            },
            node_name="explanation",
            timeout=90.0,
        )
        for line in exp_lines:
            yield line

        if exp_result is None:
            state["error_log"].append("WARNING: Explanation Agent failed")
        else:
            state["final_output"] = exp_result

    # ── Assemble final response ────────────────────────────────────────────────

    elapsed = round(time.perf_counter() - start_time, 2)
    final_output = state.get("final_output") or {}

    final_response = {
        "patient_id":             patient_id,
        "analysis_timestamp":     datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":        elapsed,
        "imaging_performed":      bool(image_data),

        "clinician_output":  final_output.get("clinician_output"),
        "patient_output":    final_output.get("patient_output"),
        "risk_attribution":  final_output.get("risk_attribution"),
        "fhir_bundle":       final_output.get("fhir_bundle"),
        "reading_level_check": final_output.get("reading_level_check"),

        "consensus": {
            "status":               consensus_result.get("consensus_status"),
            "aggregate_confidence": consensus_result.get("aggregate_confidence"),
            "human_review_required": state["human_review_required"],
            "conflict_count":       consensus_result.get("conflict_count", 0),
            "summary":              consensus_result.get("consensus_summary"),
        },

        "agent_outputs": {
            "diagnosis":    state.get("diagnosis_output"),
            "lab":          state.get("lab_output"),
            "imaging":      state.get("imaging_output"),
            "drug_safety":  state.get("drug_safety_output"),
            "digital_twin": state.get("digital_twin_output"),
        },

        "error_log":        state["error_log"],
        "meditwin_version": "1.2.0",
        "streaming":        True,
    }

    yield evt_final(final_response)
    yield sse_done()