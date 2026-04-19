from __future__ import annotations
 
import asyncio
import os
import sys
import importlib
from typing import Optional, AsyncIterator, Callable, Awaitable
 
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
 
from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, sse_done,
)

# ════════════════════════════════════════════════════════════════════════════════
# CONSENSUS  (/stream on port 8007)
# ════════════════════════════════════════════════════════════════════════════════
 
consensus_router = APIRouter()
 
 
class ConsensusStreamRequest(BaseModel):
    diagnosis_output:   Optional[dict] = None
    lab_output:         Optional[dict] = None
    imaging_output:     Optional[dict] = None
    drug_safety_output: Optional[dict] = None
    patient_state:      Optional[dict] = None
 
 
async def _consensus_stream(req: ConsensusStreamRequest) -> AsyncIterator[str]:
    node  = "consensus"
    timer = Timer()

    # Step 1 — kick off
    yield evt_status(node, "Starting conflict detection across all agent outputs...",
                     step=1, total=3)

    from main import run_consensus

    # Step 2 — running the pipeline (this is where the work actually happens,
    # including the tiebreaker RAG call if needed)
    yield evt_status(node, "Running conflict detection and tiebreaker resolution...",
                     step=2, total=3)

    try:
        result = await asyncio.to_thread(
            run_consensus,
            req.diagnosis_output, req.lab_output,
            req.imaging_output,   req.drug_safety_output,
            req.patient_state,
        )
    except Exception as exc:
        yield evt_error(node, f"Consensus pipeline failed: {exc}", fatal=True)
        return

    status    = result.get("consensus_status", "UNKNOWN")
    conflicts = result.get("conflict_count", 0)
    conf      = result.get("aggregate_confidence", 0)
    resolution = result.get("resolution")

    # Step 3 — summarize what happened
    if conflicts == 0:
        progress_msg = f"No conflicts detected — FULL_CONSENSUS at {conf:.0%} aggregate confidence"
    elif status == "CONFLICT_RESOLVED":
        method = resolution.get("resolution_method", "TIEBREAKER_RAG") if resolution else "TIEBREAKER_RAG"
        resolved_dx = resolution.get("resolved_diagnosis_display", "") if resolution else ""
        progress_msg = (
            f"{conflicts} conflict(s) resolved via {method}. "
            f"Final diagnosis: {resolved_dx} ({conf:.0%} confidence)"
        )
    elif status == "ESCALATION_REQUIRED":
        escalation_flag = result.get("escalation_flag") or {}
        priority = escalation_flag.get("priority", "MODERATE")
        # Distinguish tiebreaker fallback from genuine high-severity escalation
        reason = escalation_flag.get("reason", "")
        if "Tiebreaker RAG unavailable" in reason:
            progress_msg = (
                f"{conflicts} conflict(s) detected. RAG resolution unavailable — "
                f"escalating to human review (priority: {priority})"
            )
        else:
            progress_msg = (
                f"{conflicts} conflict(s) detected — escalating to human review "
                f"(priority: {priority})"
            )
    else:
        progress_msg = f"Consensus complete: {status}"

    yield evt_progress(node, progress_msg, pct=90)

    # Step 3 complete — emit final result
    yield evt_status(node, "Building final consensus output...", step=3, total=3)
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@consensus_router.post("/stream")
async def consensus_stream(request: ConsensusStreamRequest):
    async def gen():
        async for chunk in _consensus_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)