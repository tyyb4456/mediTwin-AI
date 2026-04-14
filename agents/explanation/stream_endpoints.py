"""
agents/explanation/stream_endpoints.py
----------------------------------------
SSE streaming endpoint for the Explanation Agent.
Mirrors agents/diagnosis/stream_endpoint.py exactly — same pattern, same events.

Add to main.py:
    from stream_endpoints import explanation_router as stream_router
    app.include_router(stream_router)

Events emitted (identical schema to diagnosis agent):
  1.  status   — "Generating SOAP note (LLM streaming)..."
  2.  token    — individual SOAP LLM tokens (streamed from Gemini, node="explanation_soap")
  3.  progress — "SOAP note complete — N tokens"
  4.  status   — "Applying medical disclaimer..."
  5.  status   — "Generating patient explanation (grade-6 gate, streaming tokens)..."
  6.  token    — individual patient LLM tokens (node="explanation_patient")
  7.  progress — "Patient text ready — grade N.N"
  8.  status   — "Reading level check..."  [retry emits another status + token stream]
  9.  status   — "Building risk attribution from XGBoost feature importances..."
  10. progress — "Risk attribution complete"
  11. status   — "Assembling FHIR R4 Bundle..."
  12. progress — "FHIR Bundle assembled — N resources"
  13. complete — full ExplanationOutput + DB persist
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from typing import Optional, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

from db import save_explanation, ExplanationRecord

explanation_router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────

class ExplanationStreamRequest(BaseModel):
    patient_state:       dict
    consensus_output:    dict
    diagnosis_output:    Optional[dict] = None
    lab_output:          Optional[dict] = None
    imaging_output:      Optional[dict] = None
    drug_safety_output:  Optional[dict] = None
    digital_twin_output: Optional[dict] = None
    chief_complaint:     str = "Not specified"


# ── SOAP streaming ─────────────────────────────────────────────────────────────

async def _stream_soap(
    patient_state:       dict,
    consensus_output:    dict,
    lab_output:          Optional[dict],
    imaging_output:      Optional[dict],
    drug_safety_output:  Optional[dict],
    digital_twin_output: Optional[dict],
    chief_complaint:     str,
) -> AsyncIterator[tuple[str, dict, int]]:
    """
    Stream SOAP note generation token by token.

    Yields: SSE strings.
    Returns via final tuple: (soap_dict, token_count) in the last yielded item.

    Design mirrors _stream_diagnose() — astream_events() on the LangChain chain,
    catching on_chat_model_stream for tokens and on_chain_end for structured output.
    """
    from soap_generator import (
        SOAP_SYSTEM, SOAP_USER, MEDICAL_DISCLAIMER,
        generate_soap_note,   # used as sync fallback
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    node = "explanation_soap"

    # ── Build the same context strings that generate_soap_note() builds ──────
    demo        = patient_state.get("demographics", {})
    age         = demo.get("age", "?")
    gender      = demo.get("gender", "unknown")
    conditions  = ", ".join(
        c.get("display", "") for c in patient_state.get("active_conditions", [])
    ) or "None documented"
    medications = ", ".join(
        f"{m.get('drug', '')} {m.get('dose', '')}".strip()
        for m in patient_state.get("medications", [])
    ) or "None"
    allergies   = ", ".join(
        f"{a.get('substance', '')} ({a.get('reaction', '')})"
        for a in patient_state.get("allergies", [])
    ) or "NKDA"
    abnormal_labs = ", ".join(
        f"{l.get('display', '')}: {l.get('value', '')} {l.get('unit', '')} [{l.get('flag', '')}]"
        for l in patient_state.get("lab_results", [])
        if l.get("flag") in ("HIGH", "LOW", "CRITICAL")
    ) or "None"
    imaging_summary = "Not performed"
    if imaging_output and not imaging_output.get("mock", False):
        pred  = imaging_output.get("model_output", {}).get("prediction", "N/A")
        conf  = imaging_output.get("model_output", {}).get("confidence", 0)
        triage = imaging_output.get("severity_assessment", {}).get("triage_label", "")
        imaging_summary = f"Chest X-ray: {pred} ({conf:.0%} confidence, {triage})"
    drug_safety_summary = "No drug safety alerts"
    if drug_safety_output:
        status  = drug_safety_output.get("safety_status", "SAFE")
        flagged = drug_safety_output.get("flagged_medications", [])
        if flagged:
            drug_safety_summary = f"UNSAFE — {', '.join(flagged[:3])} flagged"
        elif status == "SAFE":
            approved = drug_safety_output.get("approved_medications", [])
            drug_safety_summary = f"SAFE — {', '.join(approved[:3]) or 'medications reviewed'}"
    final_diagnosis = consensus_output.get("final_diagnosis", "Diagnosis pending")
    confidence      = consensus_output.get("aggregate_confidence", 0.0)
    treatment_recommendation = "Clinical judgment required"
    if digital_twin_output and not digital_twin_output.get("mock", False):
        rec_id    = digital_twin_output.get("simulation_summary", {}).get("recommended_option", "")
        scenarios = digital_twin_output.get("scenarios", [])
        rec       = next((s for s in scenarios if s.get("option_id") == rec_id), None)
        if rec:
            treatment_recommendation = (
                f"Option {rec_id}: {rec.get('label', '')}. "
                f"Predicted 7-day recovery: {rec.get('predictions', {}).get('recovery_probability_7d', 0):.0%}"
            )
    mortality_risk   = 0.0
    readmission_risk = 0.0
    if digital_twin_output and not digital_twin_output.get("mock", False):
        baseline = digital_twin_output.get("simulation_summary", {}).get("baseline_risks", {})
        mortality_risk   = baseline.get("mortality_30d", 0.0)
        readmission_risk = baseline.get("readmission_30d", 0.0)

    invoke_vars = {
        "age": age, "gender": gender,
        "chief_complaint": chief_complaint,
        "conditions": conditions, "medications": medications, "allergies": allergies,
        "abnormal_labs": abnormal_labs, "imaging_summary": imaging_summary,
        "drug_safety_summary": drug_safety_summary,
        "final_diagnosis": final_diagnosis, "confidence": confidence,
        "treatment_recommendation": treatment_recommendation,
        "mortality_risk": mortality_risk, "readmission_risk": readmission_risk,
    }

    api_key     = os.getenv("GOOGLE_API_KEY")
    soap_result = None
    token_count = 0

    if api_key:
        llm    = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SOAP_SYSTEM),
            ("human", SOAP_USER),
        ])
        # Stream raw text; parse JSON manually at the end (same as diagnosis agent)
        chain = prompt | llm

        full_text = ""
        try:
            async for event in chain.astream_events(invoke_vars, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        yield evt_token(node, chunk.content)
                        full_text += chunk.content

                elif kind == "on_chain_end":
                    # full_text is now complete — parse JSON
                    pass

        except Exception as exc:
            yield evt_error(node, f"SOAP LLM streaming failed: {exc} — using sync fallback")
            full_text = ""

        # ── Parse collected text ───────────────────────────────────────────────
        if full_text:
            try:
                clean = re.sub(r"```(?:json)?", "", full_text).strip().rstrip("`").strip()
                soap_result = json.loads(clean)
            except json.JSONDecodeError:
                yield evt_error(node, "SOAP JSON parse failed — using sync fallback")
                soap_result = None

    # Sync fallback (no API key or parse failure)
    if soap_result is None:
        try:
            soap_result = await asyncio.to_thread(
                generate_soap_note,
                patient_state, consensus_output,
                lab_output, imaging_output, drug_safety_output, digital_twin_output,
                chief_complaint,
            )
        except Exception as exc:
            yield evt_error(node, f"SOAP sync fallback failed: {exc}")
            soap_result = {
                "subjective": f"Error: {exc}",
                "objective":  "See individual agent outputs",
                "assessment": "Clinical review required" + MEDICAL_DISCLAIMER,
                "plan":       ["Physician review required"],
                "clinical_summary_one_liner": "SOAP generation error",
            }

    # Always append disclaimer — never LLM-dependent
    soap_result["assessment"] = soap_result.get("assessment", "") + (
        "" if MEDICAL_DISCLAIMER in soap_result.get("assessment", "") else MEDICAL_DISCLAIMER
    )

    # Yield the soap result as a structured event so caller can capture it
    # We use a special internal marker event — caller filters on it
    yield json.dumps({"__soap__": soap_result, "__soap_tokens__": token_count})


# ── Patient writer streaming ───────────────────────────────────────────────────

async def _stream_patient_explanation(
    patient_state:       dict,
    consensus_output:    dict,
    drug_safety_output:  Optional[dict],
    digital_twin_output: Optional[dict],
    imaging_output:      Optional[dict],
    attempt:             int = 0,
) -> AsyncIterator[tuple]:
    """
    Stream patient explanation generation token by token.
    Mirrors diagnosis _stream_diagnose() pattern exactly.
    Handles reading level gate with up to 2 retries — each retry re-streams tokens.
    """
    from patient_writer import (
        PATIENT_SYSTEM, PATIENT_USER, STRICTER_PATIENT_USER,
        _check_reading_level, _extract_all_text, _fallback_patient_explanation,
        TARGET_GRADE, MAX_ACCEPTABLE_GRADE, MAX_RETRIES,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from shared.models import DiagnosisOutput

    node    = "explanation_patient"
    api_key = os.getenv("GOOGLE_API_KEY")

    # Build plain-language context (same logic as patient_writer.generate_patient_explanation)
    demo         = patient_state.get("demographics", {})
    age          = demo.get("age", "unknown")
    final_dx     = consensus_output.get("final_diagnosis", "your condition")
    condition_plain = final_dx.split("—")[-1].strip() if "—" in final_dx else final_dx

    findings_parts = []
    labs      = patient_state.get("lab_results", [])
    crit_labs = [l for l in labs if l.get("flag") in ("CRITICAL", "HIGH")][:3]
    if crit_labs:
        findings_parts.append(
            "Some of your blood test results were abnormal: "
            + ", ".join(f"{l.get('display', '')} was {l.get('flag', '').lower()}" for l in crit_labs)
        )
    if imaging_output and not imaging_output.get("mock", False):
        pred = imaging_output.get("model_output", {}).get("prediction", "")
        if pred == "PNEUMONIA":
            findings_parts.append("Your chest X-ray showed signs of a lung infection")
    key_findings = ". ".join(findings_parts) or "Your test results helped us understand what is wrong"

    treatment_plain = "We will give you medicine to help you get better"
    if digital_twin_output and not digital_twin_output.get("mock", False):
        rec_id    = digital_twin_output.get("simulation_summary", {}).get("recommended_option", "")
        scenarios = digital_twin_output.get("scenarios", [])
        rec       = next((s for s in scenarios if s.get("option_id") == rec_id), None)
        if rec:
            treatment_plain = f"The best plan for you is: {rec.get('label', 'treatment')}"

    special_parts = []
    if drug_safety_output:
        flagged   = drug_safety_output.get("flagged_medications", [])
        allergies = patient_state.get("allergies", [])
        if flagged and allergies:
            allergy = allergies[0].get("substance", "")
            special_parts.append(
                f"Because you are allergic to {allergy}, we have chosen medicines that are safe for you"
            )
        if drug_safety_output.get("critical_interactions", []):
            special_parts.append(
                "Some of your current medicines need extra monitoring while you are treated"
            )
    special_concerns = ". ".join(special_parts) or "No special medication concerns"

    invoke_vars = {
        "condition_name": condition_plain,
        "key_findings":   key_findings,
        "treatment_plan": treatment_plain,
        "special_concerns": special_concerns,
        "age": age,
    }

    patient_result = None
    token_count    = 0

    if api_key:
        llm           = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
        user_template = STRICTER_PATIENT_USER if attempt > 0 else PATIENT_USER
        prompt        = ChatPromptTemplate.from_messages([
            ("system", PATIENT_SYSTEM),
            ("human", user_template),
        ])
        chain  = prompt | llm
        full_text = ""

        try:
            async for event in chain.astream_events(invoke_vars, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        yield evt_token(node, chunk.content)
                        full_text += chunk.content

        except Exception as exc:
            yield evt_error(node, f"Patient LLM streaming failed (attempt {attempt+1}): {exc}")
            full_text = ""

        if full_text:
            try:
                clean = re.sub(r"```(?:json)?", "", full_text).strip().rstrip("`").strip()
                patient_result = json.loads(clean)
            except json.JSONDecodeError:
                yield evt_error(node, f"Patient JSON parse failed (attempt {attempt+1})")
                patient_result = None

    # Sync fallback
    if patient_result is None:
        patient_result, _ = _fallback_patient_explanation(
            condition_plain, key_findings, treatment_plain, special_concerns, age
        )

    # Reading level gate — mirrors patient_writer.py logic
    all_text      = _extract_all_text(patient_result)
    reading_stats = _check_reading_level(all_text)
    reading_stats["attempts"] = attempt + 1

    if not reading_stats["acceptable"] and attempt < MAX_RETRIES:
        yield evt_status(
            node,
            f"Reading grade {reading_stats['grade_level']:.1f} too high (max {MAX_ACCEPTABLE_GRADE}) "
            f"— retrying with stricter prompt (attempt {attempt + 2})...",
            step=attempt + 2,
            total=MAX_RETRIES + 1,
        )
        async for chunk in _stream_patient_explanation(
            patient_state, consensus_output, drug_safety_output,
            digital_twin_output, imaging_output, attempt=attempt + 1,
        ):
            yield chunk
        return

    # Yield structured result for caller to capture
    yield json.dumps({
        "__patient__": patient_result,
        "__patient_tokens__": token_count,
        "__reading_stats__": reading_stats,
    })


# ── Core streaming generator ───────────────────────────────────────────────────

async def _stream_explanation(req: ExplanationStreamRequest) -> AsyncIterator[str]:
    """
    Full streaming pipeline for the Explanation Agent.
    Mirrors agents/diagnosis/stream_endpoint.py _stream_diagnose() exactly.
    """
    from fhir_bundler import build_fhir_bundle
    from soap_generator import MEDICAL_DISCLAIMER

    node    = "explanation"
    timer   = Timer()
    request_id = str(uuid.uuid4())[:8]
    patient_id = req.patient_state.get("patient_id", "unknown")

    # ── Step 1: SOAP note with token streaming ─────────────────────────────────
    yield evt_status(node,
                     "Generating SOAP note (streaming Gemini tokens)...",
                     step=1, total=5)

    soap_result = None
    soap_tokens = 0

    async for chunk in _stream_soap(
        req.patient_state, req.consensus_output,
        req.lab_output, req.imaging_output,
        req.drug_safety_output, req.digital_twin_output,
        req.chief_complaint,
    ):
        # Internal result marker — don't forward to client
        if chunk.startswith("{") and '"__soap__"' in chunk:
            try:
                data        = json.loads(chunk)
                soap_result = data["__soap__"]
                soap_tokens = data.get("__soap_tokens__", 0)
            except Exception:
                pass
        else:
            yield chunk  # SSE line → forward to client

    yield evt_progress(
        node,
        f"SOAP note complete — {soap_tokens} tokens streamed",
        pct=25,
    )

    # ── Step 2: Medical disclaimer (deterministic — never LLM) ────────────────
    yield evt_status(node, "Appending medical disclaimer (deterministic)...", step=2, total=5)
    if soap_result:
        soap_result["assessment"] = soap_result.get("assessment", "") + (
            "" if MEDICAL_DISCLAIMER in soap_result.get("assessment", "") else MEDICAL_DISCLAIMER
        )
    yield evt_progress(node, "Medical disclaimer appended", pct=32)

    # ── Step 3: Patient explanation with token streaming ──────────────────────
    yield evt_status(
        node,
        "Generating patient explanation at grade-6 reading level (streaming tokens)...",
        step=3, total=5,
    )

    patient_result = None
    patient_tokens = 0
    reading_stats  = {"grade_level": 6.0, "acceptable": True, "attempts": 1}

    async for chunk in _stream_patient_explanation(
        req.patient_state, req.consensus_output,
        req.drug_safety_output, req.digital_twin_output, req.imaging_output,
        attempt=0,
    ):
        if chunk.startswith("{") and '"__patient__"' in chunk:
            try:
                data           = json.loads(chunk)
                patient_result = data["__patient__"]
                patient_tokens = data.get("__patient_tokens__", 0)
                reading_stats  = data.get("__reading_stats__", reading_stats)
            except Exception:
                pass
        else:
            yield chunk  # forward to client

    grade    = reading_stats.get("grade_level", "?")
    attempts = reading_stats.get("attempts", 1)
    yield evt_progress(
        node,
        f"Patient explanation ready — grade {grade} | {patient_tokens} tokens | {attempts} attempt(s)",
        pct=60,
    )

    # ── Step 4: Risk attribution ───────────────────────────────────────────────
    yield evt_status(
        node,
        "Building risk attribution from XGBoost feature importances...",
        step=4, total=5,
    )

    try:
        # Import at call time to avoid circular imports
        from main import build_risk_attribution, extract_risk_flags
        risk_attribution = await asyncio.to_thread(
            build_risk_attribution,
            req.digital_twin_output,
            req.consensus_output,
        )
        risk_flags = extract_risk_flags(
            req.patient_state, req.drug_safety_output, req.lab_output
        )
    except Exception as exc:
        yield evt_error(node, f"Risk attribution failed (non-fatal): {exc}")
        risk_attribution = {
            "readmission_risk_explanation": "Attribution unavailable",
            "shap_style_breakdown": [],
            "model_note": str(exc),
        }
        risk_flags = []

    yield evt_progress(node, "Risk attribution complete", pct=75)

    # ── Step 5: FHIR Bundle ───────────────────────────────────────────────────
    yield evt_status(node, "Assembling FHIR R4 Bundle...", step=5, total=5)

    try:
        fhir_bundle = await asyncio.to_thread(
            build_fhir_bundle,
            patient_id,
            req.diagnosis_output,
            req.imaging_output,
            req.drug_safety_output,
            req.digital_twin_output,
        )
    except Exception as exc:
        yield evt_error(node, f"FHIR bundle assembly failed (non-fatal): {exc}")
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [],
            "error": str(exc),
        }

    entry_count    = fhir_bundle.get("_entry_count", len(fhir_bundle.get("entry", [])))
    resource_types = fhir_bundle.get("_resource_types", [])
    yield evt_progress(
        node,
        f"FHIR Bundle assembled — {entry_count} resources ({', '.join(resource_types) or 'none'})",
        pct=92,
    )

    # ── Assemble final result ─────────────────────────────────────────────────
    if patient_result:
        patient_result["reading_level_check"] = reading_stats

    result = {
        "clinician_output": {
            "soap_note":                  soap_result or {},
            "clinical_summary_one_liner": (soap_result or {}).get("clinical_summary_one_liner", ""),
            "risk_flags":                 risk_flags,
            "confidence_breakdown": {
                "overall":          req.consensus_output.get("aggregate_confidence", 0.0),
                "consensus_status": req.consensus_output.get("consensus_status", "UNKNOWN"),
                "conflict_count":   req.consensus_output.get("conflict_count", 0),
            },
        },
        "patient_output":        patient_result or {},
        "risk_attribution":      risk_attribution,
        "fhir_bundle":           fhir_bundle,
        "reading_level_check":   reading_stats,
        "consensus_status":      req.consensus_output.get("consensus_status", "UNKNOWN"),
        "human_review_required": req.consensus_output.get("human_review_required", False),
    }

    elapsed = timer.elapsed_ms()

    # ── Persist to DB ──────────────────────────────────────────────────────────
    await save_explanation(ExplanationRecord(
        request_id=request_id,
        patient_id=patient_id,
        chief_complaint=req.chief_complaint,
        consensus_status=req.consensus_output.get("consensus_status", "UNKNOWN"),
        final_diagnosis=req.consensus_output.get("final_diagnosis", ""),
        aggregate_confidence=req.consensus_output.get("aggregate_confidence", 0.0),
        human_review_required=req.consensus_output.get("human_review_required", False),
        soap_note=soap_result or {},
        patient_output=patient_result or {},
        risk_attribution=risk_attribution,
        fhir_bundle_summary={
            "entry_count":    entry_count,
            "resource_types": resource_types,
        },
        risk_flags=risk_flags,
        reading_grade_level=reading_stats.get("grade_level", 0.0),
        reading_acceptable=reading_stats.get("acceptable", True),
        reading_attempts=reading_stats.get("attempts", 1),
        soap_tokens=soap_tokens,
        patient_tokens=patient_tokens,
        source="stream",
        elapsed_ms=elapsed,
        cache_hit=False,
    ))

    yield evt_complete(node, result, elapsed_ms=elapsed)


# ── Route ──────────────────────────────────────────────────────────────────────

@explanation_router.post("/stream")
async def explanation_stream(request: ExplanationStreamRequest):
    """
    SSE streaming version of /explain.
    Streams LLM tokens from both SOAP note and patient explanation generation.
    """
    async def generator():
        async for chunk in _stream_explanation(request):
            yield chunk
        yield sse_done()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )