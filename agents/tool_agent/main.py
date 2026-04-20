"""
Agent 10: MediTwin Tool Agent
Port: 8010

A conservative LangGraph ReAct agent. The LLM decides whether to call tools:
  - Query with patient ID  → fetches context, calls only relevant tools
  - Query without patient ID → answers from general medical knowledge, no tools

Endpoints:
  POST /query  — Natural language clinical query (the only entry point)
  GET  /health — Health + tool registry status
  GET  /.well-known/agent-card — A2A metadata
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent import build_tool_agent
from stream_endpoint import router as stream_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("meditwin.tool_agent")

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver 

# ── Global agent instance ──────────────────────────────────────────────────────
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_uri = os.getenv(
        "POSTGRES_CHECKPOINT_URI",
        "postgresql://postgres:postgres@postgres-checkpoint:5432/meditwin_checkpoints"
    )

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:  
        global _agent
        logger.info("MediTwin Tool Agent starting...")
        _agent = await build_tool_agent(checkpointer)
        logger.info("✓ MediTwin Tool Agent ready on port 8010")
        yield
        logger.info("✓ MediTwin Tool Agent shutdown")


app = FastAPI(
    title="MediTwin Tool Agent",
    description=(
        "Conservative ReAct agent. Answers general medical questions directly. "
        "Calls specialist agent tools only when a patient ID is present and relevant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(stream_router)


# ── Request model ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    """
    Natural language clinical query. Examples:

    With patient ID (tools will be called as needed):
      "What is the diagnosis for patient example?"
      "Are the labs normal for patient-123?"
      "Is it safe to give amoxicillin to patient example?"
      "Generate a SOAP note for patient example."

    Without patient ID (answered from knowledge, no tools):
      "What is the mechanism of action of azithromycin?"
      "What are the diagnostic criteria for community-acquired pneumonia?"
      "What is the difference between WBC and CRP as infection markers?"
    """
    session_id: Optional[str] = None
    """
    Optional session ID for thread-scoped memory.
    If omitted, defaults to 'default'. Use patient ID here for continuity
    across multiple queries about the same patient.
    """


# ── Response helpers ───────────────────────────────────────────────────────────

def _extract_final_answer(messages: list) -> str:
    """Return the last AI text response that contains no tool calls."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not getattr(msg, "tool_calls", None) and isinstance(msg.content, str) and msg.content:
                return msg.content
    return ""


def _extract_tools_called(messages: list) -> list[str]:
    """Return deduplicated list of tool names called, in order."""
    seen = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name and name not in seen:
                    seen.append(name)
    return seen


def _extract_tool_outputs(messages: list) -> dict:
    """Return last output per tool as parsed dict (for structured consumers)."""
    import json
    # Build a map from tool_call_id → tool name via AIMessage.tool_calls
    id_to_name: dict = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tc_id and tc_name:
                    id_to_name[tc_id] = tc_name

    outputs = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = id_to_name.get(getattr(msg, "tool_call_id", ""), "unknown")
            try:
                outputs[name] = json.loads(msg.content)
            except Exception:
                outputs[name] = msg.content
    return outputs


# ── /query — single entry point ───────────────────────────────────────────────

@app.post("/query")
async def query(request: QueryRequest) -> JSONResponse:
    """
    Natural language clinical query endpoint.

    The agent decides autonomously:
      - If a patient ID is found → fetches patient context, then calls only
        the tools relevant to what the query is asking
      - If no patient ID → answers directly from medical knowledge, no tools

    The session_id field enables conversation continuity (memory).
    Pass the same session_id across multiple related queries.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    session_id = request.session_id or "default"
    config = {"configurable": {"thread_id": session_id}}

    start_time = time.time()
    logger.info(f"Query received — session={session_id} | query={request.query[:100]}")

    try:
        final_state = await _agent.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config,
        )
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    elapsed = round(time.time() - start_time, 2)
    messages = final_state.get("messages", [])
    tools_called = _extract_tools_called(messages)
    answer = _extract_final_answer(messages)
    tool_outputs = _extract_tool_outputs(messages) if tools_called else {}

    mode = "patient_specific" if tools_called else "general_knowledge"
    logger.info(f"Query complete — {elapsed}s | mode={mode} | tools={tools_called}")

    return JSONResponse(content={
        "answer":         answer,
        "mode":           mode,
        "tools_called":   tools_called,
        "tool_outputs":   tool_outputs,
        "elapsed_seconds": elapsed,
        "session_id":     session_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    })


# ── /health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    from tools import MEDITWIN_TOOLS
    return JSONResponse(content={
        "status":          "healthy" if _agent else "initializing",
        "mode":            "conservative_react",
        "tools_available": [t.name for t in MEDITWIN_TOOLS],
        "tool_count":      len(MEDITWIN_TOOLS),
        "memory_enabled":  True,
        "version":         "1.0.0",
    })


# ── A2A Agent Card ─────────────────────────────────────────────────────────────

@app.get("/.well-known/agent-card")
async def agent_card() -> JSONResponse:
    from tools import MEDITWIN_TOOLS
    tools = MEDITWIN_TOOLS
    return JSONResponse(content={
        "name":    "MediTwin Tool Agent",
        "version": "1.0.0",
        "port":    8010,
        "type":    "Conservative ReAct Tool-Calling Agent",
        "description": (
            "A LangGraph ReAct agent where all 8 MediTwin specialist agents are @tools. "
            "The LLM triages every query: if a patient ID is present it fetches patient "
            "context then selectively calls only the relevant specialist tools; "
            "if no patient ID is present it answers from general medical knowledge with no tool calls."
        ),
        "triage_logic": {
            "with_patient_id":    "fetch_patient_context → selective tools based on query intent",
            "without_patient_id": "direct LLM answer from medical knowledge, zero tool calls",
        },
        "vs_orchestrator": {
            "orchestrator_8000": "Deterministic graph — always runs all 8 agents in fixed order",
            "tool_agent_8010":   "LLM-driven triage — minimal tools, zero tools for general questions",
        },
        "tools": [{"name": t.name, "description": t.description.split("WHEN TO USE:")[0].strip()} for t in tools],
        "capabilities": [
            "natural_language_query",
            "automatic_patient_id_detection",
            "selective_tool_invocation",
            "general_medical_knowledge",
            "thread_scoped_memory",
            "conservative_tool_calling",
        ],
        "endpoint": {
            "query": "POST /query — single natural language entry point",
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)