"""
Agent 9: MediTwin Orchestrator — with Streaming Support
Port: 8000

New endpoints:
  POST /analyze         — original JSON response (unchanged)
  POST /analyze/stream  — Server-Sent Events (SSE) real-time stream

SSE stream emits:
  data: {"type": "status",   "node": "...", "message": "..."}   ← node started
  data: {"type": "result",   "node": "...", "summary": {...}}    ← node finished with data
  data: {"type": "error",    "node": "...", "message": "..."}   ← node failed
  data: {"type": "complete", "node": "...", "summary": {...}}    ← final node done
  data: {"type": "llm_token","node": "...", "token": "..."}      ← LLM token (if streamed)
  data: {"type": "final",    "data": {...}}                       ← complete JSON result
  data: [DONE]                                                    ← stream end
"""
import os
import sys
import time
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streaming_graph import stream_full_analysis
from shared.sse_utils import SSE_HEADERS

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from graph import build_meditwin_graph_with_checkpointer
from agent_callers import (
    PATIENT_CONTEXT_URL, DIAGNOSIS_URL, LAB_ANALYSIS_URL,
    DRUG_SAFETY_URL, IMAGING_TRIAGE_URL, DIGITAL_TWIN_URL,
    CONSENSUS_URL, EXPLANATION_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("meditwin.orchestrator")

_graph = None


# ── Request models ─────────────────────────────────────────────────────────────
class MediTwinRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    fhir_base_url: Optional[str] = "https://hapi.fhir.org/baseR4"
    image_data: Optional[str] = None
    sharp_token: Optional[str] = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver 

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_uri = os.getenv(
        "POSTGRES_CHECKPOINT_URI",
        "postgresql://postgres:postgres@postgres-checkpoint:5432/meditwin_checkpoints"
    )

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        global _graph
        logger.info("MediTwin Orchestrator starting...")
        _graph = await build_meditwin_graph_with_checkpointer(checkpointer)
        logger.info("✓ LangGraph StateGraph compiled with PostgreSQL checkpointer + streaming")
        logger.info("✓ MediTwin Orchestrator ready on port 8000")
        yield
        logger.info("✓ MediTwin Orchestrator shutdown")


app = FastAPI(
    title="MediTwin AI Orchestrator",
    description=(
        "Multi-agent clinical decision support system with real-time streaming. "
        "Use POST /analyze for full JSON response or POST /analyze/stream for SSE."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared: build initial state ────────────────────────────────────────────────
def _build_initial_state(
    patient_id: str,
    chief_complaint: str,
    fhir_base_url: str,
    sharp_token: str,
    image_data: Optional[str],
) -> dict:
    return {
        "patient_id":          patient_id,
        "chief_complaint":     chief_complaint,
        "fhir_base_url":       fhir_base_url,
        "sharp_token":         sharp_token,
        "image_data":          image_data,
        "imaging_available":   bool(image_data),
        "patient_state":       None,
        "diagnosis_output":    None,
        "lab_output":          None,
        "drug_safety_output":  None,
        "imaging_output":      None,
        "digital_twin_output": None,
        "consensus_output":    None,
        "final_output":        None,
        "human_review_required": False,
        "error_log":           [],
    }


def _build_final_response(final_state: dict, patient_id: str, elapsed: float, has_image: bool) -> dict:
    final_output = final_state.get("final_output") or {}
    return {
        "patient_id":        patient_id,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":   elapsed,
        "imaging_performed": has_image,

        "clinician_output":    final_output.get("clinician_output"),
        "patient_output":      final_output.get("patient_output"),
        "risk_attribution":    final_output.get("risk_attribution"),
        "fhir_bundle":         final_output.get("fhir_bundle"),
        "reading_level_check": final_output.get("reading_level_check"),

        "consensus": {
            "status":               final_state.get("consensus_output", {}).get("consensus_status"),
            "aggregate_confidence": final_state.get("consensus_output", {}).get("aggregate_confidence"),
            "human_review_required": final_state.get("human_review_required", False),
            "conflict_count":       final_state.get("consensus_output", {}).get("conflict_count", 0),
            "summary":              final_state.get("consensus_output", {}).get("consensus_summary"),
        },

        "agent_outputs": {
            "diagnosis":    final_state.get("diagnosis_output"),
            "lab":          final_state.get("lab_output"),
            "imaging":      final_state.get("imaging_output"),
            "drug_safety":  final_state.get("drug_safety_output"),
            "digital_twin": final_state.get("digital_twin_output"),
        },

        "error_log":       final_state.get("error_log", []),
        "meditwin_version": "1.1.0",
        "memory_enabled":  True,
    }


# ── POST /analyze — original JSON endpoint (unchanged behavior) ────────────────
@app.post("/analyze")
async def analyze(
    request: MediTwinRequest,
    x_sharp_patient_id:   Optional[str] = Header(None),
    x_sharp_fhir_token:   Optional[str] = Header(None),
    x_sharp_fhir_base_url: Optional[str] = Header(None),
) -> JSONResponse:
    start_time = time.time()

    patient_id    = x_sharp_patient_id    or request.patient_id
    sharp_token   = x_sharp_fhir_token    or request.sharp_token or ""
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"

    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not request.chief_complaint:
        raise HTTPException(status_code=400, detail="chief_complaint is required")

    logger.info(f"Analysis request: patient={patient_id}, imaging={'yes' if request.image_data else 'no'}")

    initial_state = _build_initial_state(
        patient_id, request.chief_complaint, fhir_base_url, sharp_token, request.image_data
    )
    config = {"configurable": {"thread_id": patient_id}}

    try:
        final_state = await _graph.ainvoke(initial_state, config)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MediTwin analysis failed: {str(e)}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Analysis complete in {elapsed}s")

    if final_state.get("patient_state") is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Patient Context Agent failed — cannot retrieve patient data",
                "patient_id": patient_id,
                "suggestion": "Check FHIR server connectivity and patient ID",
                "error_log": final_state.get("error_log", []),
                "elapsed_seconds": elapsed,
            },
        )

    return JSONResponse(content=_build_final_response(
        final_state, patient_id, elapsed, bool(request.image_data)
    ))


# ── Health check ───────────────────────────────────────────────────────────────
AGENT_ENDPOINTS = {
    "patient_context": f"{PATIENT_CONTEXT_URL}/health",
    "diagnosis":       f"{DIAGNOSIS_URL}/health",
    "lab_analysis":    f"{LAB_ANALYSIS_URL}/health",
    "drug_safety":     f"{DRUG_SAFETY_URL}/health",
    "imaging_triage":  f"{IMAGING_TRIAGE_URL}/health",
    "digital_twin":    f"{DIGITAL_TWIN_URL}/health",
    "consensus":       f"{CONSENSUS_URL}/health",
    "explanation":     f"{EXPLANATION_URL}/health",
}

@app.post("/analyze/stream")
async def analyze_stream(
    request: MediTwinRequest,
    x_sharp_patient_id:    Optional[str] = Header(None),
    x_sharp_fhir_token:    Optional[str] = Header(None),
    x_sharp_fhir_base_url: Optional[str] = Header(None),
) -> StreamingResponse:
    
    patient_id    = x_sharp_patient_id    or request.patient_id
    sharp_token   = x_sharp_fhir_token    or request.sharp_token or ""
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not request.chief_complaint:
        raise HTTPException(status_code=400, detail="chief_complaint is required")
    return StreamingResponse(
        stream_full_analysis(patient_id, request.chief_complaint, fhir_base_url, sharp_token, request.image_data),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

async def _check_agent(name: str, url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return {"status": "healthy", "details": r.json()}
            return {"status": "unhealthy", "http_status": r.status_code}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


@app.get("/health")
async def health() -> JSONResponse:
    results = await asyncio.gather(*[
        _check_agent(name, url)
        for name, url in AGENT_ENDPOINTS.items()
    ])
    agent_health = dict(zip(AGENT_ENDPOINTS.keys(), results))
    healthy_count = sum(1 for v in agent_health.values() if v["status"] == "healthy")

    checkpointer_status = "unknown"
    try:
        if _graph and hasattr(_graph, "checkpointer"):
            checkpointer_status = "active"
        else:
            checkpointer_status = "not configured"
    except Exception:
        checkpointer_status = "error"

    return JSONResponse(content={
        "status": "healthy" if healthy_count == len(AGENT_ENDPOINTS) else "degraded",
        "orchestrator": "healthy",
        "agents_healthy": healthy_count,
        "agents_total": len(AGENT_ENDPOINTS),
        "agents": agent_health,
        "graph_compiled": _graph is not None,
        "checkpointer": checkpointer_status,
        "memory_enabled": checkpointer_status == "active",
        "streaming_enabled": True,
        "stream_endpoint": "POST /analyze/stream",
        "version": "1.1.0",
    })


# ── A2A Agent Card ─────────────────────────────────────────────────────────────
@app.get("/.well-known/agent-card")
async def agent_card() -> JSONResponse:
    return JSONResponse(content={
        "name": "MediTwin AI",
        "version": "1.1.0",
        "description": (
            "Multi-agent clinical decision support system with real-time streaming. "
            "8 specialist AI agents producing differential diagnosis, lab interpretation, "
            "imaging triage (CNN), drug safety (FDA), digital twin outcome simulation, "
            "and plain-language patient communication — all FHIR R4 compliant."
        ),
        "author": "Tayyab Hussain",
        "tagline": "What is happening? What will happen next? What should we do?",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id":      {"type": "string"},
                "chief_complaint": {"type": "string"},
                "image_data":      {"type": "string", "description": "Base64 chest X-ray (optional)"},
            },
            "required": ["patient_id", "chief_complaint"],
        },
        "sharp_context": {
            "consumes": ["patient_id", "fhir_token", "fhir_base_url"],
            "required": False,
        },
        "capabilities": [
            "differential_diagnosis",
            "lab_interpretation",
            "drug_interaction_checking",
            "medical_imaging_analysis_cnn",
            "outcome_simulation_digital_twin",
            "consensus_conflict_detection",
            "fhir_r4_compliant_output",
            "patient_plain_language_explanation",
            "short_term_memory_persistence",
            "real_time_sse_streaming",
        ],
        "endpoints": {
            "analyze":        "POST /analyze         — full JSON response",
            "analyze_stream": "POST /analyze/stream  — SSE real-time stream",
            "health":         "GET  /health",
            "agent_card":     "GET  /.well-known/agent-card",
        },
        "agents": [
            {"id": 1, "name": "Patient Context Agent",      "port": 8001, "type": "A2A"},
            {"id": 2, "name": "Diagnosis Agent",            "port": 8002, "type": "A2A"},
            {"id": 3, "name": "Lab Analysis Agent",         "port": 8003, "type": "A2A"},
            {"id": 4, "name": "Drug Safety MCP Agent",      "port": 8004, "type": "MCP+A2A"},
            {"id": 5, "name": "Imaging Triage Agent",       "port": 8005, "type": "A2A"},
            {"id": 6, "name": "Digital Twin Agent",         "port": 8006, "type": "A2A"},
            {"id": 7, "name": "Consensus+Escalation Agent", "port": 8007, "type": "LangGraph node"},
            {"id": 8, "name": "Explanation Agent",          "port": 8009, "type": "A2A"},
        ],
        "mcp_superpower": {
            "name":     "drug-safety-mcp",
            "endpoint": f"{DRUG_SAFETY_URL}/mcp/",
            "tools":    ["check_drug_interactions", "get_contraindications", "suggest_alternatives"],
        },
        "memory": {
            "type":           "short_term",
            "implementation": "PostgreSQL checkpointer",
            "scope":          "thread_scoped (per patient_id)",
        },
        "streaming": {
            "protocol":    "Server-Sent Events (SSE)",
            "endpoint":    "POST /analyze/stream",
            "event_types": ["status", "result", "error", "complete", "llm_token", "final"],
            "modes":       ["messages", "updates", "custom"],
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)