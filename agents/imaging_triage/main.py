"""
Agent 5: Imaging Triage Agent — v2.1
CNN-based chest X-ray analysis — wraps trained EfficientNetB0 model.
Port: 8005

Changes in v2.1:
  - REMOVED: LLM synthetic fallback mode (no image → orchestrator does not route here)
  - ADDED:   LLM interpretation layer after CNN inference — structured clinical opinion
             mirrors the diagnosis agent's astream_events pattern
  - FIXED:   _model_loaded import-by-value bug → now uses is_model_loaded() accessor
  - Swagger UI: replaced raw base64 field with upload-friendly hint

Key design decisions (preserved):
  - Model loaded ONCE at startup via lifespan — never per-request
  - TF inference runs in thread pool (CPU-bound) — event loop never blocked
  - Preprocessing EXACTLY matches Kaggle training pipeline (raw [0-255], no /255)
  - FHIR DiagnosticReport always includes AI disclaimer
  - imaging_available=False in PatientState → orchestrator skips this agent
"""
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inference import (
    load_model_from_disk,
    is_model_loaded,          # ← accessor, not bare bool (fixes import-by-value bug)
    get_model_error,          # ← same fix
    decode_base64_image,
    preprocess_image,
    run_inference,
    classify_severity,
    build_fhir_diagnostic_report,
    mock_inference,
    MODEL_PATH,
    PNEUMONIA_THRESHOLD,
)

import db
from db import init as db_init, close as db_close, save_imaging_result, ImagingRecord
from history_router import router as history_router
from stream_endpoint import imaging_router


# ── LLM Interpretation ────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as PydanticBaseModel

class ImagingLLMInterpretation(PydanticBaseModel):
    """Structured LLM interpretation of CNN model output."""
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
        ),
        description="Mandatory disclaimer appended to all LLM interpretations."
    )


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
- Be concise and clinically precise"""

_LLM_HUMAN = """CNN MODEL OUTPUT:
- Prediction:           {prediction}
- Pneumonia Probability: {pneumonia_prob:.1%}
- Normal Probability:    {normal_prob:.1%}
- Model Confidence:      {confidence:.1%}
- Triage Grade:          {grade}
- Triage Priority:       {priority_label} (P{priority})
- Clinical Urgency:      {clinical_urgency}

IMAGING FINDINGS (rule-based):
- Pattern:            {pattern}
- Affected Area:      {affected_area}
- Bilateral:          {bilateral}
- Finding Confidence: {confidence_in_findings}

PATIENT CONTEXT:
- Age:              {age}y
- Gender:           {gender}
- Chief Complaint:  {chief_complaint}
- Working Diagnosis:{current_diagnosis}

Interpret these findings and provide structured clinical guidance."""


async def run_llm_interpretation(
    prediction: str,
    pneumonia_prob: float,
    normal_prob: float,
    confidence: float,
    severity: dict,
    imaging_findings: dict,
    patient_context,
) -> Optional[ImagingLLMInterpretation]:
    """Call Gemini with structured output to interpret CNN results."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
    structured_llm = llm.with_structured_output(ImagingLLMInterpretation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _LLM_SYSTEM),
        ("human",  _LLM_HUMAN),
    ])

    chain = prompt | structured_llm

    try:
        result = await chain.ainvoke({
            "prediction":           prediction,
            "pneumonia_prob":       pneumonia_prob,
            "normal_prob":          normal_prob,
            "confidence":           confidence,
            "grade":                severity["grade"],
            "priority_label":       severity["triage_label"],
            "priority":             severity["triage_priority"],
            "clinical_urgency":     severity["clinical_urgency"],
            "pattern":              imaging_findings["pattern"],
            "affected_area":        imaging_findings["affected_area"],
            "bilateral":            imaging_findings.get("bilateral", False),
            "confidence_in_findings": imaging_findings.get("confidence_in_findings", ""),
            "age":                  patient_context.age if patient_context else "unknown",
            "gender":               patient_context.gender if patient_context else "unknown",
            "chief_complaint":      (patient_context.chief_complaint or "Not specified") if patient_context else "Not specified",
            "current_diagnosis":    (patient_context.current_diagnosis or "Not specified") if patient_context else "Not specified",
        })
        return result
    except Exception as e:
        print(f"⚠️  LLM interpretation failed: {e}")
        return None


# ── Request / Response Models ──────────────────────────────────────────────────

class ImageData(BaseModel):
    format: str = Field(default="base64", description="Image encoding format")
    content_type: str = Field(default="image/jpeg", description="MIME type")
    data: str = Field(description="Base64-encoded image data (use /upload-xray for file upload)")


class PatientContext(BaseModel):
    age: int = Field(default=40)
    gender: str = Field(default="unknown")
    chief_complaint: Optional[str] = None
    current_diagnosis: Optional[str] = None


class ImagingRequest(BaseModel):
    patient_id: str = Field(default="unknown")
    image_data: ImageData
    patient_context: Optional[PatientContext] = None


class ImagingResponse(BaseModel):
    model_output: dict
    severity_assessment: dict
    imaging_findings: dict
    clinical_interpretation: str
    llm_interpretation: Optional[dict] = None   # ← NEW: structured LLM opinion
    confirms_diagnosis: bool
    diagnosis_code: Optional[str]
    recommended_actions: list[str]
    fhir_diagnostic_report: Optional[dict]
    model_loaded: bool
    mock: bool = False
    analysis_mode: str = "cnn"
    request_id: str = ""


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CNN model + init DB at startup."""
    print("Imaging Triage Agent v2.1 starting...")

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        success = await loop.run_in_executor(pool, load_model_from_disk)

    if success:
        print("✓ EfficientNetB0 model loaded and warmed up")
    else:
        print(f"⚠️  Model not loaded ({get_model_error()}) — mock mode active")

    await db_init()
    print("✓ Imaging Triage Agent v2.1 ready")

    yield

    await db_close()
    print("✓ Imaging Triage Agent shutdown")


app = FastAPI(
    title="MediTwin Imaging Triage Agent",
    description=(
        "CNN-based chest X-ray analysis (EfficientNetB0) with LLM clinical interpretation. "
        "Use POST /upload-xray to submit an image file directly (no base64 required). "
        "Use POST /analyze-xray for programmatic base64 access. "
        "Synthetic fallback is disabled — orchestrator only routes here when imaging_available=True."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(imaging_router)
app.include_router(history_router, prefix="/history", tags=["history"])


# ── File upload endpoint (replaces raw base64 in Swagger) ─────────────────────

from fastapi import UploadFile, File, Form
import base64 as b64lib

@app.post("/upload-xray", response_model=ImagingResponse, tags=["imaging"])
async def upload_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG, PNG, DICOM-derived)"),
    patient_id: str = Form(default="unknown"),
    age: int = Form(default=40),
    gender: str = Form(default="unknown"),
    chief_complaint: str = Form(default=""),
    current_diagnosis: str = Form(default=""),
):
    """
    Upload a chest X-ray image file directly — no base64 encoding required.
    This endpoint is intended for Swagger UI / manual testing.
    The orchestrator uses POST /analyze-xray with base64 data programmatically.
    """
    raw = await file.read()
    b64_data = b64lib.b64encode(raw).decode("utf-8")

    patient_context = PatientContext(
        age=age,
        gender=gender,
        chief_complaint=chief_complaint or None,
        current_diagnosis=current_diagnosis or None,
    )

    req = ImagingRequest(
        patient_id=patient_id,
        image_data=ImageData(
            format="base64",
            content_type=file.content_type or "image/jpeg",
            data=b64_data,
        ),
        patient_context=patient_context,
    )
    return await analyze_xray(req)


# ── Imaging Findings Interpreter ──────────────────────────────────────────────

def interpret_imaging_findings(pneumonia_prob: float, severity: dict) -> dict:
    if pneumonia_prob >= 0.90:
        return {
            "pattern": "Lobar or segmental consolidation",
            "affected_area": "One or more lobes (right lower lobe most common)",
            "bilateral": False,
            "confidence_in_findings": "HIGH",
        }
    elif pneumonia_prob >= 0.75:
        return {
            "pattern": "Patchy consolidation or airspace opacity",
            "affected_area": "Focal area, likely one lobe",
            "bilateral": False,
            "confidence_in_findings": "MODERATE",
        }
    elif pneumonia_prob >= PNEUMONIA_THRESHOLD:
        return {
            "pattern": "Possible early infiltrate or subtle opacity",
            "affected_area": "Indeterminate — clinical correlation required",
            "bilateral": False,
            "confidence_in_findings": "LOW",
        }
    else:
        return {
            "pattern": "No significant consolidation detected",
            "affected_area": "N/A",
            "bilateral": False,
            "confidence_in_findings": "HIGH",
        }


def build_clinical_interpretation(
    prediction: str,
    confidence: float,
    pneumonia_prob: float,
    severity: dict,
    patient_context: Optional[PatientContext],
) -> str:
    age_str = f"{patient_context.age}-year-old " if patient_context else ""
    chief = (
        f" ({patient_context.chief_complaint})"
        if patient_context and patient_context.chief_complaint
        else ""
    )
    AI_DISCLAIMER_SHORT = "AI-generated — not a substitute for radiologist review."

    if prediction == "PNEUMONIA":
        return (
            f"Chest X-ray pattern for {age_str}patient{chief} is "
            f"{'highly ' if confidence >= 0.90 else ''}consistent with pneumonia "
            f"(AI confidence {confidence:.1%}). "
            f"{severity['clinical_urgency']} {AI_DISCLAIMER_SHORT}"
        )
    else:
        return (
            f"Chest X-ray does not show significant consolidation for {age_str}patient{chief} "
            f"(normal probability {1 - pneumonia_prob:.1%}). "
            f"Clinical presentation should guide further workup. {AI_DISCLAIMER_SHORT}"
        )


def build_recommended_actions(
    prediction: str,
    severity: dict,
    patient_context: Optional[PatientContext],
) -> list[str]:
    actions = []
    if prediction == "PNEUMONIA":
        if severity["triage_priority"] <= 2:
            actions.append("Start empirical antibiotic therapy per local guidelines")
            actions.append("Blood cultures x2 before antibiotic administration")
        actions.append("Repeat chest X-ray at 48-72 hours to assess response")
        actions.append("Monitor oxygen saturation (target SpO2 >94%)")
        actions.append("Ensure adequate hydration and rest")
        if patient_context and patient_context.age >= 65:
            actions.append("Elderly patient — lower threshold for hospitalization")
    else:
        actions.append("Clinical correlation required — normal X-ray does not exclude early pneumonia")
        actions.append("Consider repeat imaging in 24-48 hours if clinical suspicion persists")
        actions.append("Monitor for symptom progression")
    actions.append("Radiologist review recommended for definitive interpretation")
    return actions


# ── Main REST Endpoint ─────────────────────────────────────────────────────────

@app.post("/analyze-xray", response_model=ImagingResponse, tags=["imaging"])
async def analyze_xray(request: ImagingRequest) -> ImagingResponse:
    """
    Analyze a chest X-ray image using the trained EfficientNetB0 CNN,
    then enrich the result with an LLM clinical interpretation.

    Accepts base64-encoded image. Returns:
      - Model prediction (PNEUMONIA / NORMAL)
      - Confidence score
      - Triage severity (1=IMMEDIATE to 4=ROUTINE)
      - LLM structured clinical interpretation (opinion, differentials, actions)
      - FHIR DiagnosticReport resource

    If no model is loaded, returns mock output clearly labeled as such.
    Results are persisted to the imaging_triage_results DB table.

    For SSE streaming, use POST /stream instead.
    For file upload (no base64), use POST /upload-xray.

    NOTE: This agent is only routed to when imaging_available=True in PatientState.
    The orchestrator does NOT send requests here without an image.
    """
    if not request.image_data or not request.image_data.data:
        raise HTTPException(status_code=400, detail="image_data.data is required")

    patient_context = request.patient_context or PatientContext()
    patient_id = request.patient_id or "unknown"
    request_id = str(uuid.uuid4())[:8]

    # ── Check model state via accessor (not stale imported bool) ─────────────
    model_loaded = is_model_loaded()

    # ── No model → mock ───────────────────────────────────────────────────────
    if not model_loaded:
        mock = mock_inference(patient_context.model_dump())

        await save_imaging_result(ImagingRecord(
            request_id=request_id,
            patient_id=patient_id,
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
            patient_age=patient_context.age,
            patient_gender=patient_context.gender,
            chief_complaint=patient_context.chief_complaint,
            mock=True,
            elapsed_ms=0,
            source="analyze-xray",
        ))

        return ImagingResponse(**mock, analysis_mode="mock", request_id=request_id)

    # ── Decode + validate ──────────────────────────────────────────────────────
    try:
        image = decode_base64_image(request.image_data.data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {e}. Ensure data is valid base64.",
        )

    if image.size[0] < 50 or image.size[1] < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({image.size[0]}x{image.size[1]}px). Minimum 50x50.",
        )

    # ── Preprocess ─────────────────────────────────────────────────────────────
    try:
        preprocessed = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    # ── CNN Inference ──────────────────────────────────────────────────────────
    try:
        pneumonia_prob = await run_inference(preprocessed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    normal_prob = 1.0 - pneumonia_prob
    prediction  = "PNEUMONIA" if pneumonia_prob >= PNEUMONIA_THRESHOLD else "NORMAL"
    confidence  = max(pneumonia_prob, normal_prob)

    severity         = classify_severity(pneumonia_prob, patient_context.age)
    imaging_findings = interpret_imaging_findings(pneumonia_prob, severity)
    clinical_interpretation = build_clinical_interpretation(
        prediction, confidence, pneumonia_prob, severity, patient_context
    )
    recommended_actions = build_recommended_actions(prediction, severity, patient_context)
    fhir_report = build_fhir_diagnostic_report(
        patient_id=patient_id,
        prediction=prediction,
        confidence=confidence,
        pneumonia_prob=pneumonia_prob,
        severity=severity,
        imaging_findings=imaging_findings,
    )

    confirms_diagnosis = prediction == "PNEUMONIA"
    diagnosis_code     = "J18.9" if confirms_diagnosis else None

    # ── LLM Interpretation (after CNN, non-blocking) ───────────────────────────
    llm_interp = await run_llm_interpretation(
        prediction=prediction,
        pneumonia_prob=pneumonia_prob,
        normal_prob=normal_prob,
        confidence=confidence,
        severity=severity,
        imaging_findings=imaging_findings,
        patient_context=patient_context,
    )
    llm_interp_dict = llm_interp.model_dump() if llm_interp else None

    # ── Persist ────────────────────────────────────────────────────────────────
    await save_imaging_result(ImagingRecord(
        request_id=request_id,
        patient_id=patient_id,
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
        pattern=imaging_findings["pattern"],
        affected_area=imaging_findings["affected_area"],
        bilateral=imaging_findings["bilateral"],
        confidence_in_findings=imaging_findings["confidence_in_findings"],
        clinical_interpretation=clinical_interpretation,
        confirms_diagnosis=confirms_diagnosis,
        diagnosis_code=diagnosis_code,
        patient_age=patient_context.age,
        patient_gender=patient_context.gender,
        chief_complaint=patient_context.chief_complaint,
        llm_enriched=llm_interp is not None,
        fhir_diagnostic_report=fhir_report,
        mock=False,
        elapsed_ms=0,
        source="analyze-xray",
    ))

    return ImagingResponse(
        model_output={
            "prediction":            prediction,
            "confidence":            round(confidence, 4),
            "pneumonia_probability": round(pneumonia_prob, 4),
            "normal_probability":    round(normal_prob, 4),
        },
        severity_assessment=severity,
        imaging_findings=imaging_findings,
        clinical_interpretation=clinical_interpretation,
        llm_interpretation=llm_interp_dict,   # ← structured LLM opinion
        confirms_diagnosis=confirms_diagnosis,
        diagnosis_code=diagnosis_code,
        recommended_actions=recommended_actions,
        fhir_diagnostic_report=fhir_report,
        model_loaded=True,
        mock=False,
        analysis_mode="cnn+llm",
        request_id=request_id,
    )


@app.get("/health")
async def health():
    model_loaded = is_model_loaded()
    return {
        "status": "healthy",
        "agent": "imaging-triage",
        "version": "2.1.0",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "model_error": get_model_error(),
        "db": "connected" if db.is_available() else "disconnected",
        "input_size": "224x224 RGB",
        "classes": ["NORMAL", "PNEUMONIA"],
        "threshold": PNEUMONIA_THRESHOLD,
        "modes": {
            "cnn+llm": "Image provided + model loaded → EfficientNetB0 inference + LLM interpretation",
            "mock": "Image provided but model not loaded → mock output",
        },
        "endpoints": {
            "upload_xray":  "POST /upload-xray (file upload, no base64 needed — use in Swagger)",
            "analyze_xray": "POST /analyze-xray (REST, base64 — used by orchestrator)",
            "stream":       "POST /stream (SSE, CNN inference with progress events)",
            "history":      "GET /history/{patient_id}",
        },
        "note": "Synthetic LLM fallback removed — orchestrator only routes here when imaging_available=True",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)