"""
Agent 5: Imaging Triage Agent
CNN-based chest X-ray analysis — wraps the user's trained pneumonia detection model.
Port: 8005

Key design decisions (from spec + strategy doc):
  - Model loaded ONCE at startup via lifespan — never per-request
  - TF inference runs in thread pool (CPU-bound) — event loop never blocked
  - Preprocessing EXACTLY matches Kaggle training pipeline
  - FHIR DiagnosticReport always includes AI disclaimer
  - Graceful degradation: no model file = clear error, not crash
  - Conditional in LangGraph: only runs if imaging_available = True

Input:  base64-encoded chest X-ray + patient context
Output: prediction, severity, FHIR DiagnosticReport
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directories to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inference import (
    load_model_from_disk,
    decode_base64_image,
    preprocess_image,
    run_inference,
    classify_severity,
    build_fhir_diagnostic_report,
    mock_inference,
    _model_loaded,
    _model_error,
    MODEL_PATH,
    PNEUMONIA_THRESHOLD,
)


# ── Request / Response Models ──────────────────────────────────────────────────

class ImageData(BaseModel):
    format: str = Field(default="base64", description="Image encoding format")
    content_type: str = Field(default="image/jpeg", description="MIME type")
    data: str = Field(description="Base64-encoded image data")


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
    confirms_diagnosis: bool
    diagnosis_code: Optional[str]
    recommended_actions: list[str]
    fhir_diagnostic_report: Optional[dict]
    model_loaded: bool
    mock: bool = False


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CNN model at startup — once, never per-request."""
    print("Imaging Triage Agent starting...")

    # Model load is synchronous (TF) — run in thread pool to not block startup
    import asyncio
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        success = await loop.run_in_executor(pool, load_model_from_disk)

    if success:
        print("✓ Imaging Triage Agent ready — CNN model loaded")
    else:
        print(f"⚠️  Imaging Triage Agent started in mock mode — {_model_error}")

    yield
    print("✓ Imaging Triage Agent shutdown")


app = FastAPI(
    title="MediTwin Imaging Triage Agent",
    description="CNN-based chest X-ray analysis for pneumonia detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Imaging Findings Interpreter ──────────────────────────────────────────────

def interpret_imaging_findings(pneumonia_prob: float, severity: dict) -> dict:
    """
    Map confidence score to structured imaging findings.
    These are pattern descriptions consistent with what the CNN is detecting.
    """
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
    """Generate a concise clinical interpretation sentence."""
    age_str = f"{patient_context.age}-year-old " if patient_context else ""
    chief = f" ({patient_context.chief_complaint})" if patient_context and patient_context.chief_complaint else ""

    if prediction == "PNEUMONIA":
        return (
            f"Chest X-ray pattern for {age_str}patient{chief} is "
            f"{'highly ' if confidence >= 0.90 else ''}consistent with pneumonia "
            f"(AI confidence {confidence:.1%}). "
            f"{severity['clinical_urgency']} "
            f"{AI_DISCLAIMER_SHORT}"
        )
    else:
        return (
            f"Chest X-ray does not show significant consolidation for {age_str}patient{chief} "
            f"(normal probability {1 - pneumonia_prob:.1%}). "
            f"Clinical presentation should guide further workup. "
            f"{AI_DISCLAIMER_SHORT}"
        )


AI_DISCLAIMER_SHORT = (
    "AI-generated — not a substitute for radiologist review."
)


def build_recommended_actions(
    prediction: str,
    severity: dict,
    patient_context: Optional[PatientContext],
) -> list[str]:
    """Return actionable clinical recommendations based on findings."""
    actions = []

    if prediction == "PNEUMONIA":
        priority = severity["triage_priority"]

        if priority <= 2:
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


# ── Main Endpoint ──────────────────────────────────────────────────────────────

@app.post("/analyze-xray", response_model=ImagingResponse)
async def analyze_xray(request: ImagingRequest) -> ImagingResponse:
    """
    Analyze a chest X-ray image using the trained pneumonia CNN.

    Accepts base64-encoded image, returns:
      - Model prediction (PNEUMONIA / NORMAL)
      - Confidence score
      - Triage severity (1=IMMEDIATE to 4=ROUTINE)
      - FHIR DiagnosticReport resource
      - Clinical recommendations

    If no model is loaded, returns mock output clearly labeled as such.
    """
    # ── Validate image input ───────────────────────────────────────────────────
    if not request.image_data or not request.image_data.data:
        raise HTTPException(status_code=400, detail="image_data.data is required")

    patient_context = request.patient_context or PatientContext()
    patient_id = request.patient_id or "unknown"

    # ── No model loaded → return mock ─────────────────────────────────────────
    if not _model_loaded:
        mock = mock_inference(patient_context.model_dump())
        return ImagingResponse(**mock)

    # ── Decode and validate image ──────────────────────────────────────────────
    try:
        image = decode_base64_image(request.image_data.data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {e}. Ensure data is valid base64."
        )

    # Minimum size check
    if image.size[0] < 50 or image.size[1] < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({image.size[0]}x{image.size[1]}px). Minimum 50x50."
        )

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        preprocessed = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    # ── Inference (async, runs in thread pool) ────────────────────────────────
    try:
        pneumonia_prob = await run_inference(preprocessed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    normal_prob = 1.0 - pneumonia_prob
    prediction = "PNEUMONIA" if pneumonia_prob >= PNEUMONIA_THRESHOLD else "NORMAL"
    confidence = max(pneumonia_prob, normal_prob)

    # ── Severity classification ───────────────────────────────────────────────
    severity = classify_severity(pneumonia_prob, patient_context.age)

    # ── Imaging findings ──────────────────────────────────────────────────────
    imaging_findings = interpret_imaging_findings(pneumonia_prob, severity)

    # ── Clinical interpretation ───────────────────────────────────────────────
    clinical_interpretation = build_clinical_interpretation(
        prediction, confidence, pneumonia_prob, severity, patient_context
    )

    # ── Recommended actions ───────────────────────────────────────────────────
    recommended_actions = build_recommended_actions(prediction, severity, patient_context)

    # ── FHIR DiagnosticReport ─────────────────────────────────────────────────
    fhir_report = build_fhir_diagnostic_report(
        patient_id=patient_id,
        prediction=prediction,
        confidence=confidence,
        pneumonia_prob=pneumonia_prob,
        severity=severity,
        imaging_findings=imaging_findings,
    )

    # ── Consensus signal ──────────────────────────────────────────────────────
    # confirms_diagnosis: True if imaging aligns with a pneumonia diagnosis
    confirms_diagnosis = prediction == "PNEUMONIA"
    diagnosis_code = "J18.9" if confirms_diagnosis else None

    return ImagingResponse(
        model_output={
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "pneumonia_probability": round(pneumonia_prob, 4),
            "normal_probability": round(normal_prob, 4),
        },
        severity_assessment=severity,
        imaging_findings=imaging_findings,
        clinical_interpretation=clinical_interpretation,
        confirms_diagnosis=confirms_diagnosis,
        diagnosis_code=diagnosis_code,
        recommended_actions=recommended_actions,
        fhir_diagnostic_report=fhir_report,
        model_loaded=True,
        mock=False,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "imaging-triage",
        "version": "1.0.0",
        "model_loaded": _model_loaded,
        "model_path": str(MODEL_PATH),
        "model_error": _model_error,
        "input_size": "224x224 RGB",
        "classes": ["NORMAL", "PNEUMONIA"],
        "threshold": PNEUMONIA_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)