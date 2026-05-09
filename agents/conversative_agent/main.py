"""
Agent 10: MediTwin Tool Agent
Port: 8010

A LangGraph ReAct agent. The LLM decides whether to call tools:
  - Query with patient ID  → fetches context, calls only relevant tools
  - Query without patient ID → answers from general medical knowledge, no tools

Endpoints:
  POST /query        — Natural language clinical query (JSON response)
  POST /query/stream — SSE streaming version
  GET  /health       — Health + tool registry status
  GET  /.well-known/agent-card — A2A metadata
"""
import os
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent import build_tool_agent
from stream_endpoint import router as stream_router
import db_reader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("meditwin.tool_agent")

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from dotenv import load_dotenv
load_dotenv()


# ── Global agent instance ──────────────────────────────────────────────────────
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    db_uri = (
        os.getenv("POSTGRES_CHECKPOINT_URI")
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/meditwin_checkpoints"
    )

    # ── Init DB reader pool (used by all tools to query stored results) ──────────
    await db_reader.init()

    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning(
            "GOOGLE_API_KEY not set — Tool Agent starting in degraded mode. "
            "Set GOOGLE_API_KEY to enable the conversational agent."
        )
        yield
        await db_reader.close()
        return

    from langgraph.checkpoint.memory import MemorySaver
    logger.info("MediTwin Tool Agent starting...")
    try:
        async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
            _agent = await build_tool_agent(checkpointer)
            logger.info("    ✔    MediTwin Tool Agent ready on port 8010")
            yield
    except Exception as e:
        logger.warning(f"    ⚠   PostgreSQL unavailable ({e}) — falling back to MemorySaver")
        try:
            checkpointer = MemorySaver()
            _agent = await build_tool_agent(checkpointer)
            logger.info("    ✔   MediTwin Tool Agent ready (MemorySaver) on port 8010")
        except Exception as e2:
            logger.error(f"Tool Agent failed to start: {e2}")
        yield

    await db_reader.close()
    logger.info("    ✔   MediTwin Tool Agent shutdown")


app = FastAPI(
    title="MediTwin Tool Agent",
    description=(
        "ReAct agent. Answers general medical questions directly. "
        "Calls specialist agent tools only when a patient ID is present and relevant."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """
    session_id: Optional[str] = None
    """
    Optional session ID for thread-scoped memory.
    If omitted, a unique session ID is auto-generated per request.
    Pass the same session_id across multiple related queries about the same patient
    to maintain conversation context.
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
    Pass the same session_id across multiple related queries. If omitted,
    a fresh unique session is created automatically.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    # Auto-generate a unique session if not provided — prevents cross-user contamination
    session_id = request.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    start_time = time.time()
    logger.info(f"Query received — session={session_id} | query={request.query[:100]}")

    try:
        final_state = await _agent.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config,
        )
    except Exception as e:
        logger.error(f"    ✘    Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    elapsed = round(time.time() - start_time, 2)
    messages = final_state.get("messages", [])
    tools_called = _extract_tools_called(messages)
    answer = _extract_final_answer(messages)
    tool_outputs = _extract_tool_outputs(messages) if tools_called else {}

    mode = "patient_specific" if tools_called else "general_knowledge"
    logger.info(f"Query complete — {elapsed}s | mode={mode} | tools={tools_called}")

    return JSONResponse(content={
        "answer":          answer,
        "mode":            mode,
        "tools_called":    tools_called,
        "tool_outputs":    tool_outputs,
        "elapsed_seconds": elapsed,
        "session_id":      session_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    })


# ── /health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    from tools import MEDITWIN_TOOLS
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    if not has_key:
        status = "degraded"
        note = "GOOGLE_API_KEY not set — conversational agent disabled"
    elif _agent:
        status = "healthy"
        note = None
    else:
        status = "initializing"
        note = None
    body = {
        "status":          status,
        "mode":            "react_tool_calling",
        "tools_available": [t.name for t in MEDITWIN_TOOLS],
        "tool_count":      len(MEDITWIN_TOOLS),
        "memory_enabled":  has_key,
        "version":         "2.0.0",
    }
    if note:
        body["note"] = note
    return JSONResponse(content=body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
