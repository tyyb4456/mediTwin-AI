"""
agents/imaging_triage/stream_endpoint.py
------------------------------------------
SSE streaming endpoint for the Imaging Triage Agent — v2.2 POLISHED

Improvements in v2.2:
  - FIXED: Same intelligent action deduplication as main.py
  - ENHANCED: Shares build_recommended_actions logic with main.py (DRY principle)
  - IMPROVED: Confidence-aware clinical interpretation

ONE MODE only (synthetic LLM fallback removed — orchestrator does not route
here without an image):

  MODE A — CNN + LLM Inference (image provided, model loaded):
    Step 1: Decode + validate base64 image
    Step 2: Preprocess (PIL → numpy, thread pool)
    Step 3: EfficientNetB0 inference (thread pool, non-blocking)
    Step 4: Severity classification + imaging findings
    Step 5: LLM streams interpretation tokens (clinical opinion, differentials,
            actions) — astream_events(v2) pattern, mirrors diagnosis agent
    Step 6: Build FHIR + Persist + complete

  MODE B — Mock (image provided but model not loaded):
    Returns mock output immediately.

Events emitted:
  status   — pipeline steps
  progress — per step completion %
  token    — live LLM tokens (Step 5)
  error    — non-fatal warnings or fatal failures
  complete — full result dict (mirrors /analyze-xray shape + llm_interpretation)
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Optional, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.sse_utils import (
    SSE_HEADERS, Timer,
    evt_status, evt_progress, evt_error, evt_complete, evt_token, sse_done,
)

from db import save_imaging_result, ImagingRecord

imaging_router = APIRouter()


# ── Structured LLM interpretation output ──────────────────────────────────────

class ImagingLLMInterpretation(BaseModel):
    """Structured LLM interpretation of CNN model output — streamed via SSE."""
    clinical_opinion: str = Field(
        description=(
            "2-3 sentence clinical opinion interpreting the CNN findings in context "
            "of the patient's age, complaint, and diagnosis. Speak as a radiologist."
        )
    )
    key_concern: str = Field(
        description="Single most important clinical concern raised by these findings."
    )
    differential: list[str] = Field(
        description="2-3 differential diagnoses to consider given the imaging result."
    )
    immediate_actions: list[str] = Field(
        description="Ordered list of 2-4 immediate clinical actions recommended."
    )
    follow_up: str = Field(
        description="Recommended follow-up imaging or tests within 24-72 hours."
    )
    safety_net: str = Field(
        description=(
            "One-sentence safety-netting advice — when to escalate urgency or "
            "return for reassessment."
        )
    )
    llm_disclaimer: str = Field(
        default=(
            "LLM clinical interpretation — AI-assisted, not a substitute for "
            "radiologist review or clinical judgment."
        )
    )


# ── LLM prompts ───────────────────────────────────────────────────────────────

_LLM_SYSTEM = """You are an experienced radiologist and clinical decision support system
integrated into MediTwin AI. You have received the output of a trained EfficientNetB0
chest X-ray triage model (AUC 0.981, Precision 0.976, Recall 0.939).

Your role is to interpret the model's numerical output in clinical context and provide
structured, actionable guidance to the treating physician.

Rules:
- Ground every statement in the model output provided — do not hallucinate findings
- Acknowledge uncertainty when confidence is moderate (50-75%)
- Always note that actual radiologist review is required for definitive interpretation
- Consider patient age when assessing risk (paediatric <5y, elderly >65y are higher risk)
- Be concise and clinically precise
- Provide practical, immediately actionable steps"""

_LLM_HUMAN = """CNN MODEL OUTPUT:
- Prediction:            {prediction}
- Pneumonia Probability: {pneumonia_prob:.1%}
- Normal Probability:    {normal_prob:.1%}
- Model Confidence:      {confidence:.1%}
- Triage Grade:          {grade}
- Triage Priority:       {priority_label} (P{priority})
- Clinical Urgency:      {clinical_urgency}

IMAGING FINDINGS (rule-based):
- Pattern:             {pattern}
- Affected Area:       {affected_area}
- Bilateral:           {bilateral}
- Finding Confidence:  {confidence_in_findings}

PATIENT CONTEXT:
- Age:               {age}y
- Gender:            {gender}
- Chief Complaint:   {chief_complaint}
- Working Diagnosis: {current_diagnosis}

Interpret these findings and provide structured clinical guidance."""


# ── Request model ──────────────────────────────────────────────────────────────

class ImagingStreamRequest(BaseModel):
    patient_id: str = Field(default="unknown")

    # Image is required — orchestrator only routes here when imaging_available=True
    image_data: Optional[dict] = Field(
        default=None,
        description="Required: {format: 'base64', content_type: 'image/jpeg', data: '<b64>'}"
    )

    # Patient context — enriches both CNN severity and LLM interpretation
    patient_context: Optional[dict] = Field(
        default=None,
        description="Optional: {age, gender, chief_complaint, current_diagnosis}"
    )

    # Full patient state — for richer LLM context
    patient_state: Optional[dict] = Field(
        default=None,
        description="Full PatientState from Patient Context Agent"
    )


# ── Helpers (v2.2 POLISHED — shared logic with main.py) ───────────────────────

def _normalize_action(action: str) -> str:
    """Normalize an action string for deduplication."""
    return action.strip().rstrip('.').lower()


def _build_actions(
    prediction: str,
    severity: dict,
    age: int,
    llm_interp: Optional[ImagingLLMInterpretation] = None,
) -> list[str]:
    """
    v2.2 POLISHED: Merge LLM + CNN actions intelligently.
    Same logic as main.py build_recommended_actions.
    """
    final_actions = []
    seen_normalized = set()
    
    # ── Step 1: LLM actions first ─────────────────────────────────────────────
    if llm_interp and llm_interp.immediate_actions:
        for action in llm_interp.immediate_actions:
            norm = _normalize_action(action)
            if norm not in seen_normalized:
                final_actions.append(action)
                seen_normalized.add(norm)
    
    # ── Step 2: CNN rule-based actions ────────────────────────────────────────
    cnn_actions = []
    if prediction == "PNEUMONIA":
        if severity["triage_priority"] <= 2:
            cnn_actions.append("Start empirical antibiotic therapy per local guidelines")
            cnn_actions.append("Blood cultures x2 before antibiotic administration")
        cnn_actions.append("Repeat chest X-ray at 48-72 hours to assess response")
        cnn_actions.append("Monitor oxygen saturation (target SpO2 >94%)")
        if age > 65:
            cnn_actions.append(f"Elderly patient ({age}y) — lower threshold for hospitalization")
        elif age < 5:
            cnn_actions.append(f"Paediatric patient ({age}y) — close monitoring required")
    else:
        cnn_actions.append("Clinical correlation required — normal X-ray does not exclude early pneumonia")
        cnn_actions.append("Consider repeat imaging in 24-48 hours if clinical suspicion persists")
        cnn_actions.append("Monitor for symptom progression")
    
    for action in cnn_actions:
        norm = _normalize_action(action)
        if norm not in seen_normalized:
            final_actions.append(action)
            seen_normalized.add(norm)
    
    # ── Step 3: Mandatory safety-netting ──────────────────────────────────────
    mandatory = ["Radiologist review recommended for definitive interpretation"]
    
    if llm_interp and llm_interp.safety_net:
        safety_norm = _normalize_action(llm_interp.safety_net)
        if safety_norm not in seen_normalized:
            mandatory.insert(0, llm_interp.safety_net)
            seen_normalized.add(safety_norm)
    
    for action in mandatory:
        norm = _normalize_action(action)
        if norm not in seen_normalized:
            final_actions.append(action)
            seen_normalized.add(norm)
    
    return final_actions


# ── Core streaming generator ───────────────────────────────────────────────────

async def _imaging_stream(req: ImagingStreamRequest) -> AsyncIterator[str]:
    from inference import (
        is_model_loaded,
        decode_base64_image,
        preprocess_image,
        run_inference,
        classify_severity,
        build_fhir_diagnostic_report,
        mock_inference,
        PNEUMONIA_THRESHOLD,
        MODEL_PATH,
    )

    node = "imaging_triage"
    timer = Timer()
    request_id = str(uuid.uuid4())[:8]

    image_provided = (
        req.image_data is not None
        and req.image_data.get("data")
    )

    # Patient context helpers
    patient_context = req.patient_context or {}
    patient_age     = patient_context.get("age", 40)
    patient_gender  = patient_context.get("gender", "unknown")
    chief_complaint = patient_context.get("chief_complaint") or patient_context.get("current_diagnosis", "")
    current_diagnosis = patient_context.get("current_diagnosis", "Not specified")

    # ── Guard: no image → reject (orchestrator should not send here) ──────────
    if not image_provided:
        yield evt_error(
            node,
            "No image provided. This agent requires a chest X-ray image. "
            "The orchestrator should only route here when imaging_available=True.",
            fatal=True,
        )
        return

    model_loaded = is_model_loaded()
    mode = "cnn+llm" if model_loaded else "mock"

    yield evt_status(
        node,
        f"Imaging Triage starting — mode: {mode.upper()} | image provided ✓",
        step=1, total=6,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MOCK (model not loaded)
    # ══════════════════════════════════════════════════════════════════════════

    if not model_loaded:
        yield evt_status(node, "  ⚠  Model not loaded — returning mock output", step=2, total=6)

        mock = mock_inference(patient_context)

        await save_imaging_result(ImagingRecord(
            request_id=request_id,
            patient_id=req.patient_id,
            analysis_mode="mock",
            model_loaded=False,
            image_provided=True,
            prediction="MOCK_NO_MODEL",
            confidence=None,
            pneumonia_probability=None,
            normal_probability=None,
            triage_grade="UNKNOWN",
            triage_priority=4,
            triage_label="ROUTINE",
            pattern="N/A — model not loaded",
            affected_area="N/A",
            bilateral=False,
            confidence_in_findings="NONE",
            clinical_interpretation=mock.get("clinical_interpretation", ""),
            confirms_diagnosis=False,
            diagnosis_code=None,
            patient_age=patient_age,
            patient_gender=patient_gender,
            chief_complaint=chief_complaint,
            mock=True,
            elapsed_ms=timer.elapsed_ms(),
            source="stream",
        ))

        mock["analysis_mode"] = "mock"
        mock["request_id"]    = request_id
        yield evt_complete(node, mock, elapsed_ms=timer.elapsed_ms())
        return

    # ══════════════════════════════════════════════════════════════════════════
    # CNN INFERENCE
    # ══════════════════════════════════════════════════════════════════════════

    yield evt_status(node, "Decoding and validating chest X-ray image...", step=2, total=6)

    try:
        image = await asyncio.to_thread(
            decode_base64_image, req.image_data["data"]
        )
    except Exception as e:
        yield evt_error(node, f"Image decode failed: {e}", fatal=True)
        return

    if image.size[0] < 50 or image.size[1] < 50:
        yield evt_error(
            node,
            f"Image too small ({image.size[0]}x{image.size[1]}px). Minimum 50x50.",
            fatal=True,
        )
        return

    yield evt_progress(node, f"Image decoded: {image.size[0]}x{image.size[1]}px ✓", pct=15)

    # ── Preprocess ────────────────────────────────────────────────────────────
    yield evt_status(node, "Preprocessing image for EfficientNetB0 (224×224 RGB)...", step=3, total=6)

    try:
        preprocessed = await asyncio.to_thread(preprocess_image, image)
    except Exception as e:
        yield evt_error(node, f"Preprocessing failed: {e}", fatal=True)
        return

    yield evt_progress(node, "Preprocessing complete — raw [0-255] pixel range preserved ✓", pct=30)

    # ── CNN Inference ─────────────────────────────────────────────────────────
    yield evt_status(node, "Running EfficientNetB0 inference (thread pool)...", step=4, total=6)

    try:
        pneumonia_prob = await run_inference(preprocessed)
    except Exception as e:
        yield evt_error(node, f"Model inference failed: {e}", fatal=True)
        return

    normal_prob = 1.0 - pneumonia_prob
    prediction  = "PNEUMONIA" if pneumonia_prob >= PNEUMONIA_THRESHOLD else "NORMAL"
    confidence  = max(pneumonia_prob, normal_prob)

    yield evt_progress(
        node,
        f"CNN inference complete — {prediction} "
        f"(confidence: {confidence:.1%}, pneumonia_prob: {pneumonia_prob:.1%}) ✓",
        pct=50,
    )

    # ── Severity + findings ───────────────────────────────────────────────────
    severity = await asyncio.to_thread(classify_severity, pneumonia_prob, patient_age)

    if pneumonia_prob >= 0.90:
        pattern       = "Lobar or segmental consolidation"
        affected_area = "One or more lobes (right lower lobe most common)"
        cif           = "HIGH"
    elif pneumonia_prob >= 0.75:
        pattern       = "Patchy consolidation or airspace opacity"
        affected_area = "Focal area, likely one lobe"
        cif           = "MODERATE"
    elif pneumonia_prob >= PNEUMONIA_THRESHOLD:
        pattern       = "Possible early infiltrate or subtle opacity"
        affected_area = "Indeterminate — clinical correlation required"
        cif           = "LOW"
    else:
        pattern       = "No significant consolidation detected"
        affected_area = "N/A"
        cif           = "HIGH"

    imaging_findings = {
        "pattern":                pattern,
        "affected_area":          affected_area,
        "bilateral":              False,
        "confidence_in_findings": cif,
    }

    # ── v2.2 POLISHED: Confidence-aware clinical interpretation ───────────────
    age_str       = f"{patient_age}-year-old " if patient_age else ""
    complaint_str = f" ({chief_complaint})" if chief_complaint else ""
    AI_DISCLAIMER = "AI-generated — not a substitute for radiologist review."

    if prediction == "PNEUMONIA":
        certainty_qualifier = ""
        if confidence < 0.75:
            certainty_qualifier = "possibly "
        elif confidence >= 0.90:
            certainty_qualifier = "highly "
        
        clinical_interpretation = (
            f"Chest X-ray pattern for {age_str}patient{complaint_str} is "
            f"{certainty_qualifier}consistent with pneumonia "
            f"(AI confidence {confidence:.1%}). "
            f"{severity['clinical_urgency']} {AI_DISCLAIMER}"
        )
    else:
        caveat = ""
        if pneumonia_prob >= 0.30:
            caveat = " However, early-stage pneumonia cannot be excluded on imaging alone. "
        
        clinical_interpretation = (
            f"Chest X-ray does not show significant consolidation for {age_str}patient{complaint_str} "
            f"(normal probability {normal_prob:.1%}).{caveat} "
            f"Clinical presentation should guide further workup. {AI_DISCLAIMER}"
        )

    confirms_diagnosis = prediction == "PNEUMONIA"
    diagnosis_code     = "J18.9" if confirms_diagnosis else None

    fhir_report = build_fhir_diagnostic_report(
        patient_id=req.patient_id,
        prediction=prediction,
        confidence=confidence,
        pneumonia_prob=pneumonia_prob,
        severity=severity,
        imaging_findings=imaging_findings,
    )

    yield evt_progress(node, "Severity classified + FHIR assembled ✓", pct=65)

    # ══════════════════════════════════════════════════════════════════════════
    # LLM INTERPRETATION (streaming tokens)
    # ══════════════════════════════════════════════════════════════════════════

    yield evt_status(
        node,
        "Streaming LLM clinical interpretation of CNN findings...",
        step=5, total=6,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
    structured_llm = llm.with_structured_output(ImagingLLMInterpretation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _LLM_SYSTEM),
        ("human",  _LLM_HUMAN),
    ])

    chain = prompt | structured_llm
    token_count = 0
    llm_interp: Optional[ImagingLLMInterpretation] = None

    llm_input = {
        "prediction":             prediction,
        "pneumonia_prob":         pneumonia_prob,
        "normal_prob":            normal_prob,
        "confidence":             confidence,
        "grade":                  severity["grade"],
        "priority_label":         severity["triage_label"],
        "priority":               severity["triage_priority"],
        "clinical_urgency":       severity["clinical_urgency"],
        "pattern":                pattern,
        "affected_area":          affected_area,
        "bilateral":              False,
        "confidence_in_findings": cif,
        "age":                    patient_age,
        "gender":                 patient_gender,
        "chief_complaint":        chief_complaint or "Not specified",
        "current_diagnosis":      current_diagnosis,
    }

    try:
        async for event in chain.astream_events(llm_input, version="v2"):
            kind = event["event"]
            name = event["name"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    token_count += 1
                    yield evt_token(node, chunk.content)

            elif kind == "on_chain_end" and name == "RunnableSequence":
                raw = event["data"].get("output")
                if isinstance(raw, ImagingLLMInterpretation):
                    llm_interp = raw
                elif isinstance(raw, dict):
                    try:
                        llm_interp = ImagingLLMInterpretation(**raw)
                    except Exception as pe:
                        yield evt_error(node, f"LLM parse failed: {pe}", fatal=False)

    except Exception as exc:
        yield evt_error(node, f"LLM streaming failed: {exc} — attempting sync fallback", fatal=False)
        try:
            llm_interp = await chain.ainvoke(llm_input)
        except Exception as exc2:
            yield evt_error(node, f"LLM fallback also failed: {exc2}", fatal=False)
            # Non-fatal — we still return CNN results without LLM interpretation

    llm_interp_dict = llm_interp.model_dump() if llm_interp else None

    yield evt_progress(
        node,
        f"LLM interpretation complete — {token_count} tokens streamed ✓",
        pct=90,
    )

    # ── v2.2 POLISHED: Build actions with intelligent merge ───────────────────
    recommended_actions = _build_actions(prediction, severity, patient_age, llm_interp)

    # ══════════════════════════════════════════════════════════════════════════
    # PERSIST + COMPLETE
    # ══════════════════════════════════════════════════════════════════════════

    yield evt_status(node, "Persisting result to database...", step=6, total=6)

    await save_imaging_result(ImagingRecord(
        request_id=request_id,
        patient_id=req.patient_id,
        analysis_mode="cnn+llm",
        model_loaded=True,
        image_provided=True,
        prediction=prediction,
        confidence=confidence,
        pneumonia_probability=pneumonia_prob,
        normal_probability=normal_prob,
        triage_grade=severity["grade"],
        triage_priority=severity["triage_priority"],
        triage_label=severity["triage_label"],
        pattern=pattern,
        affected_area=affected_area,
        bilateral=False,
        confidence_in_findings=cif,
        clinical_interpretation=clinical_interpretation,
        confirms_diagnosis=confirms_diagnosis,
        diagnosis_code=diagnosis_code,
        patient_age=patient_age,
        patient_gender=patient_gender,
        chief_complaint=chief_complaint,
        llm_enriched=llm_interp is not None,
        llm_token_count=token_count,
        fhir_diagnostic_report=fhir_report,
        mock=False,
        elapsed_ms=timer.elapsed_ms(),
        source="stream",
    ))

    result = {
        "model_output": {
            "prediction":            prediction,
            "confidence":            round(confidence, 4),
            "pneumonia_probability": round(pneumonia_prob, 4),
            "normal_probability":    round(normal_prob, 4),
        },
        "severity_assessment":    severity,
        "imaging_findings":       imaging_findings,
        "clinical_interpretation": clinical_interpretation,
        "llm_interpretation":     llm_interp_dict,   # ← structured LLM opinion
        "confirms_diagnosis":     confirms_diagnosis,
        "diagnosis_code":         diagnosis_code,
        "recommended_actions":    recommended_actions,  # ← v2.2 POLISHED merge
        "fhir_diagnostic_report": fhir_report,
        "model_loaded":           True,
        "mock":                   False,
        "analysis_mode":          "cnn+llm",
        "llm_token_count":        token_count,
        "request_id":             request_id,
    }

    yield evt_progress(node, "Analysis complete ✓", pct=100)
    yield evt_complete(node, result, elapsed_ms=timer.elapsed_ms())


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@imaging_router.post("/stream", tags=["imaging"])
async def imaging_stream(request: ImagingStreamRequest):
    """
    SSE streaming endpoint for imaging triage — v2.2 POLISHED.

    Requires an image (orchestrator only routes here when imaging_available=True).

    MODE A (image provided + model loaded):
      Streams: decode → CNN inference → severity → LLM interpretation tokens → FHIR → complete
      The LLM layer streams clinical opinion, differentials, and actions in real time.

    MODE B (image provided but no model file):
      Returns mock output immediately with clear instructions.

    Events: status | progress | token | error | complete
    
    v2.2 IMPROVEMENTS:
      - Intelligent LLM + CNN action deduplication
      - Confidence-aware clinical interpretation
      - Shared logic with main.py (DRY principle)
    """
    async def gen():
        async for chunk in _imaging_stream(request):
            yield chunk
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)