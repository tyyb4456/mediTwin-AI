"""
agents/conversative_agent/conversations_router.py
--------------------------------------------------
REST endpoints for managing chat conversation history stored in PostgreSQL.

Endpoints:
  GET    /conversations            — list all conversations (newest first)
  POST   /conversations            — create a new conversation
  PUT    /conversations/{id}       — save/update messages for a conversation
  DELETE /conversations/{id}       — delete a conversation
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import db_reader

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    id: str
    session_id: str


class UpdateConversationRequest(BaseModel):
    messages: list


@router.get("")
async def list_conversations():
    """Return all conversations ordered by most recently updated."""
    convs = await db_reader.list_conversations()
    return JSONResponse(content={"conversations": convs})


@router.post("")
async def create_conversation(req: CreateConversationRequest):
    """Create a new empty conversation."""
    ok = await db_reader.create_conversation(req.id, req.session_id)
    if not ok:
        raise HTTPException(status_code=503, detail="Database unavailable — conversation not saved")
    return JSONResponse(content={"status": "created", "id": req.id})


@router.put("/{conv_id}")
async def update_conversation(conv_id: str, req: UpdateConversationRequest):
    """Save the current messages array for a conversation."""
    clean = [
        {k: v for k, v in m.items() if k != "streaming"}
        for m in req.messages
        if not m.get("streaming", False)
    ]
    ok = await db_reader.update_conversation_messages(conv_id, clean)
    if not ok:
        raise HTTPException(status_code=503, detail="Database unavailable — messages not saved")
    return JSONResponse(content={"status": "updated", "id": conv_id, "message_count": len(clean)})


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    ok = await db_reader.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(status_code=503, detail="Database unavailable — conversation not deleted")
    return JSONResponse(content={"status": "deleted", "id": conv_id})
