"""
agents/diagnosis/stream_endpoint.py
-------------------------------------
SSE streaming endpoint for the Diagnosis Agent.

Add to agents/diagnosis/main.py:

    from stream_endpoint import router as stream_router
    app.include_router(stream_router)

Events emitted:
  1. status   — "Building patient query..."
  2. status   — "Retrieving clinical guidelines from ChromaDB..." (RAG mode)
     OR status — "Running LLM-only diagnosis (fallback mode)..."
  3. progress — per RAG retrieval step
  4. status   — "Running LLM differential diagnosis..."
  5. token    — individual LLM tokens (streamed from Gemini)
  6. status   — "Applying clinical rules (allergy filter, sepsis flag)..."
  7. complete — full DiagnosisOutput + FHIR conditions
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

router = APIRouter()


class StreamDiagnoseRequest(BaseModel):
    patient_state: dict
    chief_complaint: str
    include_fhir_resources: bool = True

    @field_validator("chief_complaint")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("chief_complaint cannot be empty")
        return v

    @field_validator("patient_state")
    @classmethod
    def has_patient_id(cls, v: dict) -> dict:
        if not v.get("patient_id"):
            raise ValueError("patient_state.patient_id is required")
        return v


async def _stream_diagnose(
    patient_state: dict,
    chief_complaint: str,
    include_fhir: bool,
) -> AsyncIterator[str]:
    """Core streaming generator for the diagnosis pipeline."""

    from rag import diagnosis  # singleton DiagnosisRAG from main module
    import uuid

    node = "diagnosis"
    timer = Timer()
    patient_id = patient_state.get("patient_id", "unknown")
    request_id = str(uuid.uuid4())[:8]

    if not diagnosis._initialized:
        yield evt_error(node, "Diagnosis Agent not initialized — ChromaDB may be down",
                        fatal=True)
        return

    # ── Step 1: Cache check ────────────────────────────────────────────────────
    from rag import _cache
    cached = _cache.get(patient_id, chief_complaint)
    if cached:
        yield evt_status(node, "Cache hit — returning cached diagnosis result")
        fhir = None
        if include_fhir:
            try:
                fhir = diagnosis.build_fhir_conditions(cached, patient_id)
            except Exception:
                pass
        yield evt_complete(node, {
            **_diagnosis_output_to_dict(cached),
            "fhir_conditions": fhir,
            "rag_mode": "rag" if diagnosis.rag_available else "fallback",
            "request_id": request_id,
            "cache_hit": True,
        }, elapsed_ms=timer.elapsed_ms())
        return

    # ── Step 2: Build patient query ────────────────────────────────────────────
    yield evt_status(node, "Building clinical patient query...", step=1, total=5)
    patient_query = diagnosis._build_patient_query(patient_state, chief_complaint)

    # ── Step 3: RAG retrieval or fallback ──────────────────────────────────────
    if diagnosis.rag_available:
        yield evt_status(node,
                         "Retrieving clinical guidelines from ChromaDB (RAG mode)...",
                         step=2, total=5)
        try:
            context = await asyncio.to_thread(
                diagnosis._retrieve_context, patient_query, 6
            )
            yield evt_progress(node, f"Retrieved clinical context ({len(context)} chars)", pct=35)
        except Exception as exc:
            yield evt_error(node, f"RAG retrieval failed: {exc} — falling back to LLM-only")
            context = "No clinical guidelines available — apply clinical judgment."
        system_prompt_key = "rag"
    else:
        yield evt_status(node, "Running LLM-only mode (ChromaDB unavailable)...",
                         step=2, total=5)
        context = "No clinical guidelines available — apply clinical judgment."
        system_prompt_key = "fallback"

    # ── Step 4: LLM inference with token streaming ─────────────────────────────
    yield evt_status(node,
                     "Running Gemini differential diagnosis (streaming tokens)...",
                     step=3, total=5)

    # We stream tokens from the LLM, then parse the complete output afterwards.
    # This gives the user live feedback that "something is happening".
    from rag import (
        SYSTEM_PROMPT, FALLBACK_SYSTEM_PROMPT, HUMAN_PROMPT,
        _validate_and_repair_items,
    )
    from langchain_core.prompts import ChatPromptTemplate
    from shared.models import DiagnosisOutput, NextStep
    import re, json

    system_prompt = SYSTEM_PROMPT if system_prompt_key == "rag" else FALLBACK_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", HUMAN_PROMPT),
    ])

    full_response = ""
    token_count = 0

    try:
        # Stream tokens from the LLM directly
        chain = prompt | diagnosis._llm  # raw LLM, no structured output parser yet
        async for chunk in chain.astream({
            "context": context,
            "patient_data": patient_query,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_response += token
                token_count += 1
                yield evt_token(node, token)

    except Exception as exc:
        yield evt_error(node, f"LLM streaming failed: {exc} — retrying non-streaming")
        # Fall back to non-streaming invoke
        try:
            result_obj = await asyncio.to_thread(
                diagnosis._invoke_llm, patient_query, context, system_prompt
            )
            full_response = result_obj.model_dump_json()
        except Exception as exc2:
            yield evt_error(node, f"LLM completely failed: {exc2}", fatal=True)
            return

    yield evt_progress(node,
                       f"LLM generated {token_count} tokens — parsing response...",
                       pct=70)

    # ── Step 5: Parse + validate LLM output ───────────────────────────────────
    yield evt_status(node, "Parsing and validating LLM output...", step=4, total=5)

    try:
        output = _parse_llm_response(full_response, diagnosis)
        if output is None:
            yield evt_error(node, "Could not parse LLM output as DiagnosisOutput", fatal=True)
            return
    except Exception as exc:
        yield evt_error(node, f"Parse error: {exc}", fatal=True)
        return

    # ── Step 6: Rule adjustments ───────────────────────────────────────────────
    yield evt_status(node,
                     "Applying clinical rules (allergy filter, sepsis flag, confidence boost)...",
                     step=5, total=5)

    output = await asyncio.to_thread(
        diagnosis._apply_rule_adjustments, output, patient_state
    )

    # Sync top fields
    if output.differential_diagnosis:
        top = output.differential_diagnosis[0]
        output.top_diagnosis    = top.display
        output.top_icd10_code   = top.icd10_code

    # Cache it
    _cache.set(patient_id, chief_complaint, output)

    # ── FHIR resources ─────────────────────────────────────────────────────────
    fhir_conditions = None
    if include_fhir:
        try:
            fhir_conditions = await asyncio.to_thread(
                diagnosis.build_fhir_conditions, output, patient_id
            )
        except Exception:
            pass

    elapsed = timer.elapsed_ms()
    output_dict = _diagnosis_output_to_dict(output)

    yield evt_complete(node, {
        **output_dict,
        "fhir_conditions": fhir_conditions,
        "rag_mode": "rag" if diagnosis.rag_available else "fallback",
        "request_id": request_id,
        "cache_hit": False,
        "summary": {
            "top_diagnosis":    output.top_diagnosis,
            "icd10_code":       output.top_icd10_code,
            "confidence":       output.confidence_level,
            "differential_n":   len(output.differential_diagnosis),
            "penicillin_alert": output.penicillin_allergy_flagged,
            "sepsis_alert":     output.high_suspicion_sepsis,
        },
    }, elapsed_ms=elapsed)


def _diagnosis_output_to_dict(output) -> dict:
    """Convert DiagnosisOutput pydantic model → plain dict for JSON serialization."""
    return {
        "differential_diagnosis":   [d.model_dump() for d in output.differential_diagnosis],
        "top_diagnosis":            output.top_diagnosis,
        "top_icd10_code":           output.top_icd10_code,
        "confidence_level":         output.confidence_level,
        "reasoning_summary":        output.reasoning_summary,
        "recommended_next_steps":   [s.model_dump() for s in output.recommended_next_steps],
        "penicillin_allergy_flagged": output.penicillin_allergy_flagged,
        "high_suspicion_sepsis":    output.high_suspicion_sepsis,
        "requires_isolation":       output.requires_isolation,
    }

from shared.models import DiagnosisOutput, NextStep

def _parse_llm_response(full_response: str, diagnosis_rag) -> "DiagnosisOutput | None":
    """
    Parse the raw LLM text output into a DiagnosisOutput.
    Tries structured parsing first, then raw JSON fallback.
    """
    import re, json
    from rag import _validate_and_repair_items

    text = full_response.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    if "differential_diagnosis" in data:
        data["differential_diagnosis"] = _validate_and_repair_items(
            data["differential_diagnosis"]
        )

    # Coerce recommended_next_steps strings → NextStep objects
    if "recommended_next_steps" in data:
        steps = []
        for s in data["recommended_next_steps"]:
            if isinstance(s, str):
                steps.append(NextStep(category="INVESTIGATION",
                                      description=s, urgency="routine"))
            elif isinstance(s, dict):
                steps.append(NextStep(**s))
        data["recommended_next_steps"] = steps

    try:
        return DiagnosisOutput(**data)
    except Exception:
        return None


@router.post("/stream")
async def stream_diagnose(request: StreamDiagnoseRequest):
    """SSE streaming version of /diagnose."""

    async def generator():
        async for chunk in _stream_diagnose(
            request.patient_state,
            request.chief_complaint,
            request.include_fhir_resources,
        ):
            yield chunk
        yield sse_done()

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers=SSE_HEADERS)