"""
Agent 9: MediTwin Orchestrator
Port: 8000 — the ONLY agent the outside world talks to directly.

Responsibilities:
  1. Accept patient analysis requests (with or without SHARP headers)
  2. Validate inputs and initialize MediTwinState
  3. Execute the LangGraph StateGraph (ainvoke)
  4. Return the final structured output
  5. Expose A2A agent card metadata for Prompt Opinion platform
  6. Health check endpoint that reports status of ALL downstream agents

From strategy doc:
  "Build the Orchestrator last — it only makes sense once all agents exist.
   Building it first leads to building an integration layer against nothing."
"""
import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from orchestrator.graph import build_meditwin_graph
from orchestrator.agent_callers import (
    PATIENT_CONTEXT_URL, DIAGNOSIS_URL, LAB_ANALYSIS_URL,
    DRUG_SAFETY_URL, IMAGING_TRIAGE_URL, DIGITAL_TWIN_URL,
    CONSENSUS_URL, EXPLANATION_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("meditwin.orchestrator")


# ── Global graph instance ─────────────────────────────────────────────────────
_graph = None


# ── Request / Response models ──────────────────────────────────────────────────

class MediTwinRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    fhir_base_url: Optional[str] = "https://hapi.fhir.org/baseR4"
    image_data: Optional[str] = None      # base64 chest X-ray (optional)
    sharp_token: Optional[str] = None     # SHARP auth token (optional direct field)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("MediTwin Orchestrator starting...")
    _graph = build_meditwin_graph()
    logger.info("✓ LangGraph StateGraph compiled and ready")
    logger.info("✓ MediTwin Orchestrator ready on port 8000")
    yield
    logger.info("✓ MediTwin Orchestrator shutdown")


app = FastAPI(
    title="MediTwin AI Orchestrator",
    description=(
        "Multi-agent clinical decision support system. "
        "Combines diagnosis, lab analysis, imaging, drug safety, "
        "digital twin simulation, and explanation into a single clinical assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Main analysis endpoint ─────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    request: MediTwinRequest,
    x_sharp_patient_id: Optional[str] = Header(None),
    x_sharp_fhir_token: Optional[str] = Header(None),
    x_sharp_fhir_base_url: Optional[str] = Header(None),
) -> JSONResponse:
    """
    Main MediTwin analysis endpoint.

    Accepts:
      - Direct request body (development / testing)
      - SHARP headers from Prompt Opinion platform (production)

    SHARP headers take priority over request body fields.

    Returns the complete MediTwin output:
      - SOAP note (clinician)
      - Patient explanation (plain language)
      - Risk attribution (SHAP-style)
      - FHIR R4 Bundle
      - Consensus status + confidence
    """
    start_time = time.time()

    # Resolve patient ID and FHIR config — SHARP headers take priority
    patient_id    = x_sharp_patient_id   or request.patient_id
    sharp_token   = x_sharp_fhir_token   or request.sharp_token or ""
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"

    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not request.chief_complaint:
        raise HTTPException(status_code=400, detail="chief_complaint is required")

    logger.info(f"Analysis request: patient={patient_id}, "
                f"imaging={'yes' if request.image_data else 'no'}")

    # Build initial state
    initial_state = {
        "patient_id":         patient_id,
        "chief_complaint":    request.chief_complaint,
        "fhir_base_url":      fhir_base_url,
        "sharp_token":        sharp_token,
        "image_data":         request.image_data,
        "imaging_available":  bool(request.image_data),
        "patient_state":      None,
        "diagnosis_output":   None,
        "lab_output":         None,
        "drug_safety_output": None,
        "imaging_output":     None,
        "digital_twin_output": None,
        "consensus_output":   None,
        "final_output":       None,
        "human_review_required": False,
        "error_log":          [],
    }

    # Execute graph
    try:
        final_state = await _graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"MediTwin analysis failed: {str(e)}"
        )

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Analysis complete in {elapsed}s — "
                f"status={final_state.get('consensus_output', {}).get('consensus_status', '?')}")

    # Check for fatal patient context failure
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

    # Build response
    final_output = final_state.get("final_output") or {}
    response = {
        "patient_id": patient_id,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "imaging_performed": bool(request.image_data),

        # Core clinical outputs from Explanation Agent
        "clinician_output":  final_output.get("clinician_output"),
        "patient_output":    final_output.get("patient_output"),
        "risk_attribution":  final_output.get("risk_attribution"),
        "fhir_bundle":       final_output.get("fhir_bundle"),
        "reading_level_check": final_output.get("reading_level_check"),

        # Consensus summary
        "consensus": {
            "status":               final_state.get("consensus_output", {}).get("consensus_status"),
            "aggregate_confidence": final_state.get("consensus_output", {}).get("aggregate_confidence"),
            "human_review_required": final_state.get("human_review_required", False),
            "conflict_count":       final_state.get("consensus_output", {}).get("conflict_count", 0),
            "summary":              final_state.get("consensus_output", {}).get("consensus_summary"),
        },

        # Individual agent outputs (for debugging / downstream consumption)
        "agent_outputs": {
            "diagnosis":    final_state.get("diagnosis_output"),
            "lab":          final_state.get("lab_output"),
            "imaging":      final_state.get("imaging_output"),
            "drug_safety":  final_state.get("drug_safety_output"),
            "digital_twin": final_state.get("digital_twin_output"),
        },

        # Execution metadata
        "error_log": final_state.get("error_log", []),
        "meditwin_version": "1.0.0",
    }

    return JSONResponse(content=response)


# ── Health check ──────────────────────────────────────────────────────────────

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


async def _check_agent(name: str, url: str) -> dict:
    """Check a single downstream agent's health endpoint."""
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
    """
    Health check — polls all 8 downstream agents concurrently.
    Reports which are up/down for demo monitoring.
    """
    results = await asyncio.gather(*[
        _check_agent(name, url)
        for name, url in AGENT_ENDPOINTS.items()
    ])

    agent_health = dict(zip(AGENT_ENDPOINTS.keys(), results))
    healthy_count = sum(1 for v in agent_health.values() if v["status"] == "healthy")
    all_healthy = healthy_count == len(AGENT_ENDPOINTS)

    return JSONResponse(content={
        "status": "healthy" if all_healthy else "degraded",
        "orchestrator": "healthy",
        "agents_healthy": healthy_count,
        "agents_total": len(AGENT_ENDPOINTS),
        "agents": agent_health,
        "graph_compiled": _graph is not None,
        "version": "1.0.0",
    })


# ── A2A Agent Card ─────────────────────────────────────────────────────────────

@app.get("/.well-known/agent-card")
async def agent_card() -> JSONResponse:
    """
    A2A Agent Card — published to Prompt Opinion Marketplace.
    Describes MediTwin's capabilities and SHARP context requirements.
    """
    return JSONResponse(content={
        "name": "MediTwin AI",
        "version": "1.0.0",
        "description": (
            "Multi-agent clinical decision support system. "
            "Combines 8 specialist AI agents to produce a comprehensive medical assessment: "
            "differential diagnosis, lab interpretation, medical imaging triage (CNN), "
            "drug safety checking (FDA data), digital twin outcome simulation, "
            "and plain-language patient communication — all FHIR R4 compliant."
        ),
        "author": "Tayyab Hussain",
        "tagline": "What is happening? What will happen next? What should we do?",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id":      {"type": "string", "description": "FHIR Patient resource ID"},
                "chief_complaint": {"type": "string", "description": "Primary reason for visit"},
                "image_data":      {"type": "string", "description": "Base64 chest X-ray (optional)"},
            },
            "required": ["patient_id", "chief_complaint"],
        },
        "sharp_context": {
            "consumes": ["patient_id", "fhir_token", "fhir_base_url"],
            "required": False,
            "description": "SHARP headers enable direct FHIR server access",
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
        ],
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
            "name": "drug-safety-mcp",
            "endpoint": f"{DRUG_SAFETY_URL}/mcp/",
            "tools": ["check_drug_interactions", "get_contraindications", "suggest_alternatives"],
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)