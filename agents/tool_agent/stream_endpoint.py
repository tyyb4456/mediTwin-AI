"""
agents/tool_agent/stream_endpoint_v2.py
--------------------------------------
SSE streaming endpoint for the MediTwin Tool Agent.

Streams two categories of events to the frontend in real-time:

  1. TOOL EVENTS (stream_mode="custom")
     Emitted by each @tool via get_stream_writer() in tools.py:
       {"type": "tool_start",    "tool": str, "message": str}
       {"type": "tool_progress", "tool": str, "message": str}
       {"type": "tool_complete", "tool": str, "message": str, "data": dict}
       {"type": "tool_error",    "tool": str, "message": str}

  2. LLM TOKENS (stream_mode="messages")
     Token-by-token output from the Gemini reasoning / final answer.
     Routed by node name — only the final answer node tokens are sent:
       {"type": "llm_token", "token": str}

  3. LIFECYCLE EVENTS
       {"type": "status",   "message": str}
       {"type": "complete", "elapsed_ms": int}
       {"type": "error",    "message": str, "fatal": bool}

ADD to agents/tool_agent/main.py:

    from stream_endpoint import router as stream_router
    app.include_router(stream_router)

Usage (frontend):
    const es = new EventSource('/query/stream');  // GET with ?query=...&session_id=...
    // OR POST to /query/stream with JSON body

    es.onmessage = (e) => {
        if (e.data === '[DONE]') { es.close(); return; }
        const event = JSON.parse(e.data);
        // handle event.type: 'tool_start' | 'tool_complete' | 'llm_token' | 'complete' ...
    };
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel

logger = logging.getLogger("meditwin.tool_agent.stream")

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


# ── Request model ──────────────────────────────────────────────────────────────

class StreamQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


# ── Message helpers (mirrors /query endpoint) ──────────────────────────────────

def _tools_called(messages: list) -> list[str]:
    seen = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if n and n not in seen:
                    seen.append(n)
    return seen


def _tool_outputs(messages: list) -> dict:
    id_to_name: dict = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tname = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tid and tname:
                    id_to_name[tid] = tname
    outputs: dict = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            n = id_to_name.get(getattr(msg, "tool_call_id", ""), "unknown")
            try:
                outputs[n] = json.loads(msg.content)
            except Exception:
                outputs[n] = msg.content
    return outputs


def _final_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if not getattr(msg, "tool_calls", None) and isinstance(msg.content, str) and msg.content:
                return msg.content
    return ""


# ── Core streaming generator ───────────────────────────────────────────────────

async def _stream_query(
    query: str,
    session_id: str,
    agent,
) -> AsyncIterator[str]:
    """
    Core SSE generator using astream_events(v='v2').

    Works with both LangChain AgentExecutor (langchain.agents.create_agent)
    and compiled LangGraph graphs — unlike astream(stream_mode=[...]) which
    is LangGraph-only.

    Events:
      on_custom_event   → custom tool_start/progress/complete from get_stream_writer()
      on_tool_start     → native tool_start (fallback)
      on_tool_end       → native tool_complete (fallback)
      on_chat_model_stream → llm_token (handles str and Gemini list-of-parts)
      on_chain_end      → captures final state for the rich complete event
    """
    t0 = time.perf_counter()

    yield _sse({"type": "status", "message": "Processing your query..."})

    config = {"configurable": {"thread_id": session_id}}
    final_messages: list = []

    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── Custom tool writer events (get_stream_writer pattern) ──────────
            if kind == "on_custom_event":
                yield _sse({
                    "type": data.get("type", "status"),
                    "tool": data.get("tool"),
                    "message": data.get("message", ""),
                    "data": data.get("data"),
                })

            # ── Native tool lifecycle events ───────────────────────────────────
            elif kind == "on_tool_start":
                yield _sse({
                    "type": "tool_start",
                    "tool": name,
                    "message": f"Calling {name}...",
                    "input": data.get("input", {}),
                })

            elif kind == "on_tool_end":
                raw_output = data.get("output", "")
                try:
                    parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                except Exception:
                    parsed = raw_output
                yield _sse({
                    "type": "tool_complete",
                    "tool": name,
                    "message": f"{name} completed",
                    "data": parsed,
                })

            # ── LLM token streaming ────────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", None)
                if not content:
                    continue
                # Handle str and Gemini list-of-parts format
                if isinstance(content, str):
                    yield _sse({"type": "llm_token", "token": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                yield _sse({"type": "llm_token", "token": text})
                        elif isinstance(part, str) and part:
                            yield _sse({"type": "llm_token", "token": part})

            # ── Capture final agent output messages ────────────────────────────
            elif kind == "on_chain_end" and name in ("AgentExecutor", "LangGraph", "agent"):
                output = data.get("output", {})
                if isinstance(output, dict):
                    msgs = output.get("messages", [])
                    if msgs:
                        final_messages = msgs

    except Exception as exc:
        logger.error(f"Stream error: {exc}", exc_info=True)
        yield _sse({"type": "error", "message": str(exc), "fatal": True})
        return

    # ── Rich complete event — mirrors the /query JSON response shape ───────────
    called = _tools_called(final_messages)
    outputs = _tool_outputs(final_messages) if called else {}
    answer = _final_answer(final_messages)
    mode = "patient_specific" if called else "general_knowledge"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    yield _sse({
        "type": "complete",
        "elapsed_ms": elapsed_ms,
        "answer": answer,
        "mode": mode,
        "tools_called": called,
        "tool_outputs": outputs,
        "session_id": session_id,
    })


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@router.post("/query/stream")
async def stream_query(request: StreamQueryRequest):
    """
    SSE streaming version of POST /query.
    """
    from main import _agent  # import singleton from main.py at call time

    if not _agent:
        async def error_gen():
            yield _sse({"type": "error", "message": "Agent not initialized", "fatal": True})
            yield _sse_done()
        return StreamingResponse(error_gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    if not request.query or not request.query.strip():
        async def bad_req():
            yield _sse({"type": "error", "message": "query must not be empty", "fatal": True})
            yield _sse_done()
        return StreamingResponse(bad_req(), media_type="text/event-stream", headers=SSE_HEADERS)

    session_id = request.session_id or "default"

    async def generator():
        async for chunk in _stream_query(request.query, session_id, _agent):
            yield chunk
        yield _sse_done()

    return StreamingResponse(generator(), media_type="text/event-stream", headers=SSE_HEADERS)