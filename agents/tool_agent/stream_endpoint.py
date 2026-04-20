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
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
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


# ── Core streaming generator ───────────────────────────────────────────────────

async def _stream_query(
    query: str,
    session_id: str,
    agent,
) -> AsyncIterator[str]:
    """
    Core SSE generator for the tool agent.

    Uses LangGraph astream() with TWO stream modes simultaneously:
      - "custom"   → tool_start / tool_complete / tool_progress events from _emit()
      - "messages" → LLM token chunks for live typewriter output

    The agent's stream_mode list is passed directly to astream().
    """
    t0 = time.perf_counter()

    yield _sse({"type": "status", "message": "Processing your query..."})

    config = {"configurable": {"thread_id": session_id}}
    input_messages = {"messages": [HumanMessage(content=query)]}

    try:
        async for stream_mode, chunk in agent.astream(
            input_messages,
            config=config,
            stream_mode=["custom", "messages"],
        ):

            # ── Custom events from tool _emit() calls ──────────────────────────
            if stream_mode == "custom":
                # chunk is the raw dict passed to get_stream_writer()(...)
                event_type = chunk.get("type", "status")
                yield _sse({
                    "type": event_type,
                    "tool": chunk.get("tool"),
                    "message": chunk.get("message", ""),
                    "data": chunk.get("data"),
                })

            # ── LLM token streaming ────────────────────────────────────────────
            elif stream_mode == "messages":
                # chunk is a tuple: (message_chunk, metadata)
                msg_chunk, metadata = chunk

                if not isinstance(msg_chunk, AIMessageChunk):
                    continue

                # Only stream tokens from the agent node (not tool result nodes)
                langgraph_node = metadata.get("langgraph_node", "")
                if langgraph_node not in ("agent", "tools", ""):
                    continue

                content = msg_chunk.content
                if not content:
                    continue

                # Skip tool-call JSON blobs — only emit readable text tokens
                if isinstance(content, str):
                    yield _sse({"type": "llm_token", "token": content})
                elif isinstance(content, list):
                    # Gemini sometimes returns content as list of parts
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                yield _sse({"type": "llm_token", "token": text})
                        elif isinstance(part, str) and part:
                            yield _sse({"type": "llm_token", "token": part})

    except Exception as exc:
        logger.error(f"Stream error: {exc}", exc_info=True)
        yield _sse({"type": "error", "message": str(exc), "fatal": True})
        return

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    yield _sse({"type": "complete", "elapsed_ms": elapsed_ms})


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@router.post("/query/stream")
async def stream_query(request: StreamQueryRequest):
    """
    SSE streaming version of POST /query.

    Emits live events:
      - tool_start / tool_progress / tool_complete / tool_error  (from specialist tools)
      - llm_token                                                  (Gemini answer tokens)
      - status / complete / error                                  (lifecycle)
      - [DONE]                                                     (end of stream)

    Frontend usage:
        const response = await fetch('/query/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query: '...', session_id: '...' })
        });
        const reader = response.body.getReader();
        // read SSE lines...
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