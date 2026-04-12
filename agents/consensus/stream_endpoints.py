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
 
    yield evt_status(node, "Starting conflict detection across all agent outputs...",
                     step=1, total=3)
 
    from main import run_consensus  # the core consensus function in consensus/main.py
 
    yield evt_status(node, "Running conflict detection rules...", step=1, total=3)
 
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
 
    if conflicts == 0:
        yield evt_progress(node, f"No conflicts — FULL_CONSENSUS at {conf:.0%}", pct=80)
    elif status == "CONFLICT_RESOLVED":
        yield evt_progress(node, f"{conflicts} conflict(s) resolved via RAG tiebreaker", pct=80)
    else:
        yield evt_progress(node,
                           f"{conflicts} conflict(s) detected — escalating to human review",
                           pct=80)
 
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())
 
 
@consensus_router.post("/stream")
async def consensus_stream(request: ConsensusStreamRequest):
    async def gen():
        async for chunk in _consensus_stream(request):
            yield chunk
        yield sse_done()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
 
 