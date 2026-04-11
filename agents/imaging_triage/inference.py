"""
Imaging Triage Inference Module
Wraps the trained EfficientNetB0 pneumonia model for FastAPI serving.

CRITICAL: Preprocessing pipeline MUST match training exactly.
Training was done on Kaggle "Chest X-Ray Images (Pneumonia)" dataset:
  - Images resized to 224x224
  - Pixel values kept as RAW [0, 255] — EfficientNetB0 has built-in preprocessing
  - include_preprocessing=True was set in EfficientNetB0 constructor
  - RGB format (3 channels)
  - Binary sigmoid output: 0.0 = NORMAL, 1.0 = PNEUMONIA

DO NOT divide by 255 — EfficientNet handles that internally.
"""
import os
import io
import base64
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed — imaging agent will run in mock mode")


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "efficientnet_b0.keras"

# Must match training config
IMAGE_SIZE = (224, 224)

# EfficientNetB0 with include_preprocessing=True handles its own normalization
# DO NOT apply manual normalization — raw [0, 255] pixels go straight in
NORMALIZE_IMAGENET = False   # NOT used — kept for reference only

PNEUMONIA_THRESHOLD = 0.50

_executor = ThreadPoolExecutor(max_workers=2)


# ── Global model state ─────────────────────────────────────────────────────────

_model = None
_model_loaded = False
_model_error: Optional[str] = None


def load_model_from_disk() -> bool:
    global _model, _model_loaded, _model_error

    if not TF_AVAILABLE:
        _model_error = "TensorFlow not installed"
        return False

    if not MODEL_PATH.exists():
        _model_error = (
            f"Model file not found: {MODEL_PATH}. "
            "Place efficientnet_b0.keras in agents/imaging_triage/models/"
        )
        print(f"⚠️  {_model_error}")
        return False

    try:
        print(f"  Loading EfficientNetB0 model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        _model_loaded = True

        # Warm up — raw [0,255] dummy input to match inference pipeline
        dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print(f"  ✓ EfficientNetB0 loaded and warmed up: {MODEL_PATH.name}")
        return True

    except Exception as e:
        _model_error = f"Model load failed: {e}"
        print(f"  ❌ {_model_error}")
        return False


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image for EfficientNetB0 inference.

    Key difference from standard CNN:
      - NO division by 255
      - EfficientNetB0 (include_preprocessing=True) rescales internally
      - Output dtype: float32, range [0, 255]
      - Shape: (1, 224, 224, 3)
    """
    # Convert to RGB — handles grayscale X-rays and RGBA
    image = image.convert("RGB")

    # Resize to training input size
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)

    # Convert to float32 — keep raw pixel range [0, 255]
    arr = np.array(image, dtype=np.float32)

    # NO normalization — EfficientNetB0 preprocesses internally
    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    return np.expand_dims(arr, axis=0)


def decode_base64_image(b64_data: str) -> Image.Image:
    """
    Decode a base64-encoded image string to PIL Image.
    Handles both raw base64 and data URLs (data:image/jpeg;base64,...).
    """
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(image_bytes))


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference_sync(preprocessed: np.ndarray) -> float:
    """Synchronous TF inference — runs in thread pool."""
    prediction = _model.predict(preprocessed, verbose=0)
    return float(np.squeeze(prediction))


async def run_inference(preprocessed: np.ndarray) -> float:
    """Async wrapper — runs TF inference in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_inference_sync, preprocessed)


# ── Severity Classification ────────────────────────────────────────────────────

TRIAGE_LABELS = {1: "IMMEDIATE", 2: "URGENT", 3: "SEMI-URGENT", 4: "ROUTINE"}

def classify_severity(pneumonia_prob: float, patient_age: int) -> dict:
    """
    Map EfficientNetB0 confidence → clinical triage priority.
    EfficientNet is more precise (0.976) so thresholds are reliable.
    """
    if pneumonia_prob >= 0.90:
        grade = "SEVERE"
        priority = 1
        urgency = "Findings highly consistent with pneumonia. Immediate treatment recommended."
    elif pneumonia_prob >= 0.75:
        grade = "MODERATE"
        priority = 2
        urgency = "Findings consistent with moderate pneumonia. Treatment within 4-6 hours."
    elif pneumonia_prob >= PNEUMONIA_THRESHOLD:
        grade = "MILD"
        priority = 3
        urgency = "Findings may indicate early or mild pneumonia. Clinical correlation required."
    else:
        grade = "NORMAL"
        priority = 4
        urgency = "No significant consolidation detected. Routine clinical assessment."

    # Age-based priority boost
    if (patient_age > 65 or patient_age < 5) and priority > 1:
        priority -= 1
        urgency += f" Priority elevated due to patient age ({patient_age}y)."

    return {
        "grade": grade,
        "triage_priority": priority,
        "triage_label": TRIAGE_LABELS[priority],
        "clinical_urgency": urgency,
    }


# ── FHIR DiagnosticReport Builder ─────────────────────────────────────────────

AI_DISCLAIMER = (
    "AI-generated triage assistance only. Not a substitute for radiologist review. "
    "For clinical decision support use under physician supervision. "
    "Model: EfficientNetB0 trained on Kaggle Chest X-Ray dataset (binary: NORMAL/PNEUMONIA). "
    "Test AUC: 0.981 | Precision: 0.976 | Recall: 0.939."
)

def build_fhir_diagnostic_report(
    patient_id: str,
    prediction: str,
    confidence: float,
    pneumonia_prob: float,
    severity: dict,
    imaging_findings: dict,
) -> dict:
    """Build FHIR R4 DiagnosticReport resource for imaging result."""
    conclusion = (
        f"AI Triage (EfficientNetB0): {prediction} "
        f"(confidence: {confidence:.1%}). "
        f"Priority: {severity['triage_label']}. "
        f"{AI_DISCLAIMER}"
    )

    report = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "24748-7",
                    "display": "Chest X-ray AP",
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "issued": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "extension": [
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/ai-confidence",
                "valueDecimal": round(confidence, 4),
            },
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/triage-priority",
                "valueInteger": severity["triage_priority"],
            },
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/model-name",
                "valueString": "EfficientNetB0",
            },
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/model-auc",
                "valueDecimal": 0.981,
            },
        ],
    }

    if prediction == "PNEUMONIA":
        report["conclusionCode"] = [
            {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": "J18.9",
                        "display": "Pneumonia, unspecified organism",
                    }
                ]
            }
        ]

    return report


# ── Mock inference ─────────────────────────────────────────────────────────────

def mock_inference(patient_context: dict) -> dict:
    """Returns mock result when no model is available."""
    return {
        "model_output": {
            "prediction": "MOCK_NO_MODEL",
            "confidence": 0.0,
            "pneumonia_probability": 0.0,
            "normal_probability": 1.0,
            "mock": True,
        },
        "severity_assessment": {
            "grade": "UNKNOWN",
            "triage_priority": 4,
            "triage_label": "ROUTINE",
            "clinical_urgency": (
                "MOCK — No model loaded. "
                "Place efficientnet_b0.keras in agents/imaging_triage/models/"
            ),
        },
        "imaging_findings": {
            "pattern": "N/A — model not loaded",
            "affected_area": "N/A",
            "bilateral": False,
            "confidence_in_findings": "NONE",
        },
        "clinical_interpretation": (
            f"MOCK OUTPUT — Model file not found at {MODEL_PATH}. "
            f"Error: {_model_error}"
        ),
        "confirms_diagnosis": False,
        "diagnosis_code": None,
        "recommended_actions": [
            f"Place trained model at: {MODEL_PATH}",
            "Run: cp /kaggle/working/saved_models/efficientnet_b0.keras agents/imaging_triage/models/",
            "Restart the imaging triage agent",
        ],
        "fhir_diagnostic_report": None,
        "model_loaded": False,
        "mock": True,
    }