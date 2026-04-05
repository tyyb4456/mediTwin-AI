"""
Imaging Triage Inference Module
Wraps the trained pneumonia CNN for FastAPI serving.

CRITICAL: Preprocessing pipeline MUST match training exactly.
Training was done on Kaggle "Chest X-Ray Images (Pneumonia)" dataset:
  - Images resized to 224x224
  - Pixel values normalized to [0, 1] (divide by 255)
  - ImageNet mean/std normalization applied
  - RGB format (3 channels)

If your model used different preprocessing (e.g. grayscale, different size),
update IMAGE_SIZE and NORMALIZE_IMAGENET below accordingly.

Model output: Binary sigmoid
  - 0.0 = NORMAL
  - 1.0 = PNEUMONIA
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

# TensorFlow import — optional, graceful degradation if not installed
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed — imaging agent will run in mock mode")


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_cnn_v1.h5"

# Must match training config
IMAGE_SIZE = (224, 224)

# ImageNet normalization (standard for VGG16/ResNet50 transfer learning)
# Set NORMALIZE_IMAGENET = False if your model only used /255 normalization
NORMALIZE_IMAGENET = True
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Inference threshold
PNEUMONIA_THRESHOLD = 0.50

# Thread pool for CPU-bound TF inference (keeps event loop unblocked)
_executor = ThreadPoolExecutor(max_workers=2)


# ── Global model state ─────────────────────────────────────────────────────────

_model = None
_model_loaded = False
_model_error: Optional[str] = None


def load_model_from_disk() -> bool:
    """
    Load the Keras model from disk.
    Called ONCE at startup — never per-request.
    Returns True on success, False on failure.
    """
    global _model, _model_loaded, _model_error

    if not TF_AVAILABLE:
        _model_error = "TensorFlow not installed"
        return False

    if not MODEL_PATH.exists():
        _model_error = (
            f"Model file not found: {MODEL_PATH}. "
            "Place your trained pneumonia_cnn_v1.h5 in agents/imaging_triage/models/"
        )
        print(f"⚠️  {_model_error}")
        return False

    try:
        print(f"  Loading CNN model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        _model_loaded = True

        # Warm up — run one dummy inference so first real request isn't slow
        dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print(f"  ✓ Model loaded and warmed up: {MODEL_PATH.name}")
        return True

    except Exception as e:
        _model_error = f"Model load failed: {e}"
        print(f"  ❌ {_model_error}")
        return False


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image for inference.
    MUST match training preprocessing exactly.

    Returns: float32 array of shape (1, 224, 224, 3)
    """
    # Convert to RGB — handles grayscale X-rays and RGBA screenshots
    image = image.convert("RGB")

    # Resize to training input size
    image = image.resize(IMAGE_SIZE, Image.LANCZOS)

    # Convert to float32 array
    arr = np.array(image, dtype=np.float32)

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Apply ImageNet normalization if model was trained with it
    if NORMALIZE_IMAGENET:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    return np.expand_dims(arr, axis=0)


def decode_base64_image(b64_data: str) -> Image.Image:
    """
    Decode a base64-encoded image string to PIL Image.
    Handles both raw base64 and data URLs (data:image/jpeg;base64,...).
    """
    # Strip data URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(image_bytes))


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference_sync(preprocessed: np.ndarray) -> float:
    """
    Synchronous inference — run in thread pool to avoid blocking event loop.
    Returns raw sigmoid output (0.0 to 1.0).
    """
    prediction = _model.predict(preprocessed, verbose=0)
    # Flatten to scalar — handles both (1,1) and (1,) output shapes
    return float(np.squeeze(prediction))


async def run_inference(preprocessed: np.ndarray) -> float:
    """Async wrapper — runs TF inference in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_inference_sync, preprocessed)


# ── Severity Classification ────────────────────────────────────────────────────

TRIAGE_LABELS = {1: "IMMEDIATE", 2: "URGENT", 3: "SEMI-URGENT", 4: "ROUTINE"}

def classify_severity(pneumonia_prob: float, patient_age: int) -> dict:
    """
    Map model confidence → clinical triage priority.

    Priority scale:
      1 = IMMEDIATE (treat now)
      2 = URGENT (within 4-6 hours)
      3 = SEMI-URGENT (within 24 hours)
      4 = ROUTINE (normal workup)

    Age modifier: patients >65 or <5 get one priority bump.
    """
    # Base classification from probability
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

    # Age-based priority boost (elderly and pediatric patients get bumped up)
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
    "Model trained on Kaggle Chest X-Ray dataset (binary: NORMAL/PNEUMONIA)."
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
        f"AI Triage: {prediction} "
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
        ],
    }

    # Add ICD-10 conclusion code only for positive findings
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


# ── Mock inference for when no model is loaded ────────────────────────────────

def mock_inference(patient_context: dict) -> dict:
    """
    Returns a realistic-looking mock result when no model is available.
    Used for development/testing without the actual .h5 file.
    Clearly labeled as MOCK in all output fields.
    """
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
            "clinical_urgency": "MOCK — No model loaded. Place pneumonia_cnn_v1.h5 in models/ directory.",
        },
        "imaging_findings": {
            "pattern": "N/A — model not loaded",
            "affected_area": "N/A",
            "bilateral": False,
            "confidence_in_findings": "NONE",
        },
        "clinical_interpretation": (
            f"MOCK OUTPUT — Model file not found at {MODEL_PATH}. "
            "This is placeholder output for development. "
            f"Error: {_model_error}"
        ),
        "confirms_diagnosis": False,
        "diagnosis_code": None,
        "recommended_actions": [
            f"Place trained model at: {MODEL_PATH}",
            "Restart the imaging triage agent",
            "Re-submit the X-ray for real inference",
        ],
        "fhir_diagnostic_report": None,
        "model_loaded": False,
        "mock": True,
    }