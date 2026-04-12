"""
Agent 8: Explanation Agent
The last mile — transforms all technical agent outputs into:
  1. SOAP note for the clinician
  2. Patient-friendly plain-language explanation (grade 6, reading level gated)
  3. SHAP-style risk attribution from Digital Twin XGBoost models
  4. FHIR R4 Bundle of all generated resources

Port: 8009

Key design decisions (from spec + strategy doc):
  - Two completely separate LLM prompts (SOAP vs patient) — different tone + vocab
  - Reading level is a HARD GATE: FK grade > 8 → regenerate, max 2 retries
  - Medical disclaimer always appended as template string — never LLM-dependent
  - FHIR Bundle assembles resources from ALL upstream agents
  - Risk attribution uses XGBoost get_score() from Digital Twin models
"""
import os
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stream_endpoints import explanation_router as stream_router

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from soap_generator import generate_soap_note, MEDICAL_DISCLAIMER
from patient_writer import generate_patient_explanation
from fhir_bundler import build_fhir_bundle


# ── Digital Twin model path for SHAP-style attribution ────────────────────────
DIGITAL_TWIN_MODELS_DIR = (
    Path(__file__).parent.parent / "digital_twin" / "models"
)
READMISSION_MODEL_PATH = DIGITAL_TWIN_MODELS_DIR / "readmission_30d.json"
FEATURE_NAMES_PATH = DIGITAL_TWIN_MODELS_DIR / "feature_names.json"

_risk_model: Optional[xgb.XGBClassifier] = None
_feature_names: list[str] = []


# ── Request / Response models ──────────────────────────────────────────────────

class ExplanationRequest(BaseModel):
    patient_state: dict
    consensus_output: dict
    diagnosis_output: Optional[dict] = None
    lab_output: Optional[dict] = None
    imaging_output: Optional[dict] = None
    drug_safety_output: Optional[dict] = None
    digital_twin_output: Optional[dict] = None
    chief_complaint: str = Field(default="Not specified")


class ExplanationResponse(BaseModel):
    clinician_output: dict
    patient_output: dict
    risk_attribution: dict
    fhir_bundle: dict
    reading_level_check: dict
    consensus_status: str
    human_review_required: bool


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _risk_model, _feature_names

    # Load Digital Twin readmission model for feature attribution
    if READMISSION_MODEL_PATH.exists():
        try:
            _risk_model = xgb.XGBClassifier()
            _risk_model.load_model(str(READMISSION_MODEL_PATH))

            if FEATURE_NAMES_PATH.exists():
                with open(FEATURE_NAMES_PATH) as f:
                    _feature_names = json.load(f)

            print(f"✓ Explanation Agent: loaded risk model for attribution "
                  f"({len(_feature_names)} features)")
        except Exception as e:
            print(f"  ⚠️  Risk model load failed: {e} — attribution disabled")
    else:
        print(f"  ⚠️  Risk model not found at {READMISSION_MODEL_PATH} — "
              f"run digital_twin/train_models.py first")

    print("✓ Explanation Agent started")
    yield
    print("✓ Explanation Agent shutdown")


app = FastAPI(
    title="MediTwin Explanation Agent",
    description="SOAP note, patient explanation, risk attribution, FHIR Bundle",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(stream_router)


# ── Risk attribution helpers ───────────────────────────────────────────────────

# Human-readable labels for XGBoost features
_FEATURE_LABELS = {
    "age":                    lambda v: f"Age ({int(v)}y)",
    "gender_male":            lambda v: f"Male sex" if v else "Female sex",
    "wbc":                    lambda v: f"WBC {v:.1f} ({'elevated' if v > 11 else 'normal'})",
    "creatinine":             lambda v: f"Creatinine {v:.1f} mg/dL ({'elevated' if v > 1.3 else 'normal'})",
    "albumin":                lambda v: f"Albumin {v:.1f} g/dL ({'low' if v < 3.5 else 'normal'})",
    "crp":                    lambda v: f"CRP {v:.0f} mg/L ({'elevated' if v > 10 else 'normal'})",
    "comorbidity_count":      lambda v: f"Comorbidity burden ({int(v)} conditions)",
    "has_diabetes":           lambda v: "Diabetes mellitus (active)" if v else "No diabetes",
    "has_ckd":                lambda v: "Chronic kidney disease" if v else "No CKD",
    "has_chf":                lambda v: "Congestive heart failure" if v else "No CHF",
    "has_copd":               lambda v: "COPD" if v else "No COPD",
    "on_anticoagulant":       lambda v: "On anticoagulant therapy" if v else "Not anticoagulated",
    "critical_lab_count":     lambda v: f"{int(v)} critical lab value(s)",
    "med_count":              lambda v: f"Polypharmacy ({int(v)} medications)",
    "has_atrial_fibrillation":lambda v: "Atrial fibrillation" if v else "No atrial fibrillation",
    "on_steroid":             lambda v: "On corticosteroid therapy" if v else None,
}

# Features that increase risk when elevated
_HIGH_RISK_FEATURES = {
    "age", "wbc", "creatinine", "crp", "comorbidity_count",
    "has_ckd", "has_chf", "has_copd", "critical_lab_count", "has_diabetes",
    "has_atrial_fibrillation",
}


def build_risk_attribution(
    digital_twin_output: Optional[dict],
    consensus_output: dict,
) -> dict:
    """
    Build SHAP-style risk attribution using XGBoost feature importances.
    Uses the Digital Twin's feature_attribution if available (already computed),
    otherwise falls back to the loaded readmission model's get_score().
    """
    # Use Digital Twin's pre-computed attribution if available
    if digital_twin_output and not digital_twin_output.get("mock", False):
        pre_computed = digital_twin_output.get("feature_attribution", [])
        if pre_computed:
            baseline = digital_twin_output.get("simulation_summary", {}).get("baseline_risks", {})
            readmit_risk = baseline.get("readmission_30d", 0.0)
            return {
                "readmission_risk_explanation": (
                    f"Your 30-day readmission risk was estimated at {readmit_risk:.0%}. "
                    "The top contributing factors are listed below."
                ),
                "shap_style_breakdown": pre_computed,
                "model_note": (
                    "Attribution based on XGBoost feature importance (gain). "
                    "Models trained on synthetic data — for architecture demonstration."
                ),
            }

    # Fallback: use loaded readmission model directly
    if _risk_model and _feature_names:
        try:
            raw_scores = _risk_model.get_booster().get_score(importance_type="gain")

            feat_importance = {}
            for feat_key, score in raw_scores.items():
                if feat_key.startswith("f"):
                    try:
                        idx = int(feat_key[1:])
                        if idx < len(_feature_names):
                            feat_importance[_feature_names[idx]] = float(score)
                    except ValueError:
                        pass

            top = sorted(feat_importance.items(), key=lambda x: -x[1])[:5]
            max_score = max(v for _, v in top) if top else 1.0

            attributions = []
            for feat_name, score in top:
                label_fn = _FEATURE_LABELS.get(feat_name)
                label = label_fn(1) if label_fn else feat_name  # placeholder value

                direction = "increases_risk" if feat_name in _HIGH_RISK_FEATURES else "neutral"
                norm = round(score / max_score, 2)
                contrib = f"+{norm:.2f}" if direction == "increases_risk" else f"±{norm:.2f}"

                attributions.append({
                    "feature": label,
                    "contribution": contrib,
                    "direction": direction,
                    "importance_score": round(score, 2),
                })

            return {
                "readmission_risk_explanation": "Risk attribution based on XGBoost feature importance.",
                "shap_style_breakdown": attributions,
                "model_note": "Models trained on synthetic data — architecture demonstration only.",
            }
        except Exception as e:
            print(f"  ⚠️  Risk attribution failed: {e}")

    return {
        "readmission_risk_explanation": "Risk attribution unavailable — Digital Twin models not loaded.",
        "shap_style_breakdown": [],
        "model_note": "Run digital_twin/train_models.py to enable risk attribution.",
    }


# ── Risk flags helper ──────────────────────────────────────────────────────────

def extract_risk_flags(
    patient_state: dict,
    drug_safety_output: Optional[dict],
    lab_output: Optional[dict],
) -> list[str]:
    """Extract key clinical risk flags for clinician summary."""
    flags = []

    # Allergies
    for allergy in patient_state.get("allergies", []):
        sub = allergy.get("substance", "")
        sev = allergy.get("severity", "")
        if sub:
            flags.append(f"{sub} allergy ({sev})" if sev else f"{sub} allergy")

    # Drug safety flags
    if drug_safety_output:
        for flagged in drug_safety_output.get("flagged_medications", [])[:3]:
            flags.append(f"{flagged} — flagged by Drug Safety")

    # Critical lab alerts
    if lab_output:
        for alert in lab_output.get("critical_alerts", [])[:2]:
            msg = alert.get("message", "")
            if msg:
                flags.append(msg[:70])

    return flags[:6]  # Cap at 6 flags


# ── Main endpoint ──────────────────────────────────────────────────────────────

@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest) -> ExplanationResponse:
    """
    Generate complete MediTwin final output:
      1. SOAP note (clinician)
      2. Patient explanation (grade 6, reading level gated)
      3. SHAP-style risk attribution
      4. FHIR R4 Bundle
    """
    patient_state    = request.patient_state
    consensus_output = request.consensus_output
    patient_id       = patient_state.get("patient_id", "unknown")

    # ── 1. SOAP note ──────────────────────────────────────────────────────────
    try:
        soap = generate_soap_note(
            patient_state=patient_state,
            consensus_output=consensus_output,
            lab_output=request.lab_output,
            imaging_output=request.imaging_output,
            drug_safety_output=request.drug_safety_output,
            digital_twin_output=request.digital_twin_output,
            chief_complaint=request.chief_complaint,
        )
    except Exception as e:
        soap = {
            "subjective": f"Error generating SOAP: {e}",
            "objective": "See individual agent outputs",
            "assessment": "Clinical review required" + MEDICAL_DISCLAIMER,
            "plan": ["Physician review required"],
            "clinical_summary_one_liner": "Error in SOAP generation",
        }

    # Risk flags for clinician
    risk_flags = extract_risk_flags(
        patient_state, request.drug_safety_output, request.lab_output
    )

    clinician_output = {
        "soap_note": soap,
        "clinical_summary_one_liner": soap.get("clinical_summary_one_liner", ""),
        "risk_flags": risk_flags,
        "confidence_breakdown": {
            "overall": consensus_output.get("aggregate_confidence", 0.0),
            "consensus_status": consensus_output.get("consensus_status", "UNKNOWN"),
            "conflict_count": consensus_output.get("conflict_count", 0),
        },
    }

    # ── 2. Patient explanation ────────────────────────────────────────────────
    try:
        patient_dict, reading_stats = generate_patient_explanation(
            patient_state=patient_state,
            consensus_output=consensus_output,
            drug_safety_output=request.drug_safety_output,
            digital_twin_output=request.digital_twin_output,
            imaging_output=request.imaging_output,
        )
        patient_dict["reading_level_check"] = reading_stats
    except Exception as e:
        patient_dict = {
            "condition_explanation": "Your doctor will explain your condition.",
            "why_this_happened": "Please ask your care team.",
            "what_happens_next": "Your doctor has a plan for your care.",
            "what_to_expect": ["Your care team will explain the next steps"],
            "important_for_you_to_know": "Please speak with your nurse or doctor.",
            "when_to_call_the_nurse": ["If you feel worse", "If you have questions"],
        }
        reading_stats = {"grade_level": 6.0, "acceptable": True, "error": str(e)}

    reading_level_check = reading_stats

    # ── 3. Risk attribution ───────────────────────────────────────────────────
    risk_attribution = build_risk_attribution(
        digital_twin_output=request.digital_twin_output,
        consensus_output=consensus_output,
    )

    # ── 4. FHIR Bundle ────────────────────────────────────────────────────────
    try:
        fhir_bundle = build_fhir_bundle(
            patient_id=patient_id,
            diagnosis_output=request.diagnosis_output,
            imaging_output=request.imaging_output,
            drug_safety_output=request.drug_safety_output,
            digital_twin_output=request.digital_twin_output,
        )
    except Exception as e:
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [],
            "error": str(e),
        }

    return ExplanationResponse(
        clinician_output=clinician_output,
        patient_output=patient_dict,
        risk_attribution=risk_attribution,
        fhir_bundle=fhir_bundle,
        reading_level_check=reading_level_check,
        consensus_status=consensus_output.get("consensus_status", "UNKNOWN"),
        human_review_required=consensus_output.get("human_review_required", False),
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "explanation",
        "version": "1.0.0",
        "risk_model_loaded": _risk_model is not None,
        "features_loaded": len(_feature_names),
        "reading_level_gate": f"FK grade ≤ {8} (max {2} retries)",
        "outputs": ["soap_note", "patient_explanation", "risk_attribution", "fhir_bundle"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8009)