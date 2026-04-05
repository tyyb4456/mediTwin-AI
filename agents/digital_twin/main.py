"""
Agent 6: Digital Twin Agent
XGBoost risk models + treatment simulation + LLM narrative.
Port: 8006

Pipeline per request:
  1. Feature engineering from PatientState (deterministic)
  2. XGBoost inference — 3 risk scores: readmission, mortality, complication
  3. Treatment simulation — apply effect multipliers for each option
  4. Select recommended option (weighted composite score)
  5. LLM narrative — 3-sentence clinical comparison (tightly constrained)
  6. FHIR CarePlan for recommended scenario
"""
import os
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from feature_engineering import engineer_features, FEATURE_NAMES
from simulator import (
    simulate_treatment,
    determine_patient_risk_profile,
    select_recommended_option,
)

# ── Model paths ────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILES = {
    "readmission_30d": MODELS_DIR / "readmission_30d.json",
    "mortality_30d":   MODELS_DIR / "mortality_30d.json",
    "complication":    MODELS_DIR / "complication.json",
}

# ── Global model registry ──────────────────────────────────────────────────────
_models: dict[str, xgb.XGBClassifier] = {}
_models_loaded = False
_models_error: Optional[str] = None

# ── LLM ───────────────────────────────────────────────────────────────────────
_llm: Optional[ChatGoogleGenerativeAI] = None
_llm_ready = False


# ── Request / Response models ──────────────────────────────────────────────────

class TreatmentOption(BaseModel):
    option_id: str = Field(description="Short identifier: 'A', 'B', 'C'")
    label: str = Field(description="Human-readable label")
    drugs: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)

class DigitalTwinRequest(BaseModel):
    patient_state: dict
    diagnosis: str = Field(default="Unknown diagnosis")
    treatment_options: list[TreatmentOption] = Field(default_factory=list)

class DigitalTwinResponse(BaseModel):
    simulation_summary: dict
    scenarios: list[dict]
    what_if_narrative: str
    fhir_care_plan: Optional[dict]
    feature_attribution: list[dict]
    models_loaded: bool
    mock: bool = False


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _models_loaded, _models_error, _llm, _llm_ready

    # Load XGBoost models
    missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing:
        _models_error = (
            f"Model files not found: {missing}. "
            "Run: python agents/digital_twin/train_models.py"
        )
        print(f"⚠️  Digital Twin: {_models_error}")
    else:
        try:
            for name, path in MODEL_FILES.items():
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                _models[name] = m
            _models_loaded = True
            print(f"✓ Digital Twin: loaded {len(_models)} XGBoost risk models")
        except Exception as e:
            _models_error = f"Model load failed: {e}"
            print(f"❌ Digital Twin: {_models_error}")

    # Init LLM
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            _llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
            _llm_ready = True
            print("✓ Digital Twin: LLM narrative ready")
        except Exception as e:
            print(f"⚠️  Digital Twin: LLM init failed ({e}) — narrative disabled")
    else:
        print("⚠️  Digital Twin: No GOOGLE_API_KEY — narrative disabled")

    yield
    print("✓ Digital Twin Agent shutdown")


app = FastAPI(
    title="MediTwin Digital Twin Agent",
    description="XGBoost risk simulation + treatment scenario comparison",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_baseline_risks(feature_vector: list[float]) -> dict:
    """Run all 3 XGBoost models and return baseline risk scores."""
    X = np.array([feature_vector], dtype=np.float32)
    return {
        "readmission_30d": float(_models["readmission_30d"].predict_proba(X)[0][1]),
        "mortality_30d":   float(_models["mortality_30d"].predict_proba(X)[0][1]),
        "complication":    float(_models["complication"].predict_proba(X)[0][1]),
    }


def get_feature_attribution(feature_dict: dict, baseline_risks: dict) -> list[dict]:
    """
    SHAP-style feature attribution using XGBoost feature importance.
    Maps top features to human-readable descriptions with directional labels.
    Uses the readmission model as the primary attribution source.
    """
    if not _models_loaded:
        return []

    model = _models["readmission_30d"]

    # get_score returns gain-based importance
    try:
        raw_scores = model.get_booster().get_score(importance_type="gain")
    except Exception:
        return []

    # Map internal feature names (f0, f1...) to human names
    feat_importance = {}
    for feat_idx_str, score in raw_scores.items():
        if feat_idx_str.startswith("f"):
            try:
                idx = int(feat_idx_str[1:])
                if idx < len(FEATURE_NAMES):
                    feat_importance[FEATURE_NAMES[idx]] = score
            except ValueError:
                pass

    if not feat_importance:
        return []

    # Sort by importance and take top 5
    top_features = sorted(feat_importance.items(), key=lambda x: -x[1])[:5]

    # Build human-readable attributions
    FEATURE_LABELS = {
        "age":               lambda v: f"Age ({int(v)}y)",
        "wbc":               lambda v: f"WBC {v:.1f} ({'elevated' if v > 11 else 'normal'})",
        "creatinine":        lambda v: f"Creatinine {v:.1f} ({'elevated' if v > 1.3 else 'normal'})",
        "albumin":           lambda v: f"Albumin {v:.1f} ({'low' if v < 3.5 else 'normal'})",
        "crp":               lambda v: f"CRP {v:.0f} ({'elevated' if v > 10 else 'normal'})",
        "comorbidity_count": lambda v: f"Comorbidity count ({int(v)})",
        "has_ckd":           lambda v: "Chronic kidney disease" if v else "No CKD",
        "has_chf":           lambda v: "Congestive heart failure" if v else "No CHF",
        "has_diabetes":      lambda v: "Diabetes mellitus" if v else "No diabetes",
        "has_copd":          lambda v: "COPD" if v else "No COPD",
        "on_anticoagulant":  lambda v: "On anticoagulant therapy" if v else "Not anticoagulated",
        "critical_lab_count":lambda v: f"{int(v)} critical lab value(s)",
    }

    attributions = []
    for feat_name, importance in top_features:
        value = feature_dict.get(feat_name, 0)
        label_fn = FEATURE_LABELS.get(feat_name)
        label = label_fn(value) if label_fn else f"{feat_name}: {value:.2f}"

        # Direction: high-risk features increase risk, protective features decrease it
        HIGH_RISK_FEATURES = {
            "age", "wbc", "creatinine", "crp", "comorbidity_count",
            "has_ckd", "has_chf", "has_copd", "critical_lab_count",
        }
        direction = "increases_risk" if feat_name in HIGH_RISK_FEATURES and value > 0 else "reduces_risk"

        # Sign contribution based on direction
        normalized = round(importance / max(feat_importance.values()), 2)
        contribution = f"+{normalized:.2f}" if direction == "increases_risk" else f"-{normalized:.2f}"

        attributions.append({
            "feature": label,
            "contribution": contribution,
            "direction": direction,
            "importance_score": round(float(importance), 2),
        })

    return attributions


def build_llm_narrative(
    patient_state: dict,
    diagnosis: str,
    scenarios: list[dict],
    recommended_option: str,
    risk_profile: str,
) -> str:
    """
    Generate a 3-sentence clinical narrative comparing scenarios.
    Tightly constrained prompt — must cite specific numbers.
    Falls back to rule-based template if LLM unavailable.
    """
    if not _llm_ready or not _llm:
        # Rule-based fallback
        best = next((s for s in scenarios if s["option_id"] == recommended_option), scenarios[0] if scenarios else {})
        no_tx = next((s for s in scenarios if s["option_id"] == "C"), None)
        demo = patient_state.get("demographics", {})
        age = demo.get("age", "?")
        gender = demo.get("gender", "patient")

        rec_label = best.get("label", recommended_option)
        rec_7d = best.get("predictions", {}).get("recovery_probability_7d", 0)
        rec_mort = best.get("predictions", {}).get("mortality_risk_30d", 0)

        narrative = (
            f"For this {age}-year-old {gender} with {diagnosis} and {risk_profile} risk profile, "
            f"{rec_label} offers the best predicted outcome with {rec_7d:.0%} 7-day recovery probability "
            f"and {rec_mort:.0%} 30-day mortality risk. "
        )
        if no_tx:
            no_mort = no_tx.get("predictions", {}).get("mortality_risk_30d", 0)
            narrative += (
                f"Without treatment, 30-day mortality risk is estimated at {no_mort:.0%}. "
            )
        narrative += "This simulation is based on XGBoost risk models — clinical judgment required."
        return narrative

    # Build scenario summary for LLM
    scenario_lines = []
    for s in scenarios:
        preds = s.get("predictions", {})
        scenario_lines.append(
            f"Option {s['option_id']} ({s['label']}): "
            f"7-day recovery {preds.get('recovery_probability_7d', 0):.0%}, "
            f"30-day mortality {preds.get('mortality_risk_30d', 0):.0%}, "
            f"readmission {preds.get('readmission_risk_30d', 0):.0%}"
        )

    demo = patient_state.get("demographics", {})
    age = demo.get("age", "?")
    gender = demo.get("gender", "patient")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical decision support system explaining a treatment simulation to a physician. "
         "Write exactly 3 sentences. Use specific percentages from the data. Be concise and clinical. "
         "End with: 'This is AI-generated decision support — clinical judgment required.'"),
        ("human",
         f"Patient: {age}y {gender}. Diagnosis: {diagnosis}. Risk profile: {risk_profile}.\n\n"
         f"Scenario comparison:\n" + "\n".join(scenario_lines) + f"\n\n"
         f"Recommended option: {recommended_option}. "
         f"Write a 3-sentence clinical narrative comparing the options and justifying the recommendation."),
    ])

    chain = prompt | _llm | StrOutputParser()
    try:
        return chain.invoke({})
    except Exception as e:
        print(f"  ⚠️  LLM narrative failed: {e}")
        return f"Digital Twin simulation complete. Recommended option: {recommended_option}. Clinical judgment required."


def build_fhir_care_plan(
    patient_id: str,
    recommended_option: TreatmentOption,
    narrative: str,
    predicted_recovery: float,
) -> dict:
    """Build FHIR R4 CarePlan resource for the recommended treatment."""
    activities = []

    for drug in recommended_option.drugs:
        activities.append({
            "detail": {
                "kind": "MedicationRequest",
                "code": {"text": drug},
                "status": "scheduled",
                "description": f"Proposed: {drug}",
            }
        })

    for intervention in recommended_option.interventions:
        activities.append({
            "detail": {
                "kind": "ServiceRequest",
                "code": {"text": intervention},
                "status": "scheduled",
                "description": f"Proposed: {intervention}",
            }
        })

    return {
        "resourceType": "CarePlan",
        "status": "active",
        "intent": "plan",
        "subject": {"reference": f"Patient/{patient_id}"},
        "title": f"AI-Recommended Treatment Plan — Option {recommended_option.option_id}: {recommended_option.label}",
        "description": narrative,
        "activity": activities,
        "note": [
            {
                "text": (
                    f"AI-generated treatment simulation. Predicted 7-day recovery probability: {predicted_recovery:.0%}. "
                    "Not a substitute for clinical judgment. Physician review and patient consent required before implementation."
                )
            }
        ],
    }


def mock_response(request: DigitalTwinRequest) -> DigitalTwinResponse:
    """Return clearly-labeled mock output when models are not loaded."""
    demo = request.patient_state.get("demographics", {})
    return DigitalTwinResponse(
        simulation_summary={
            "patient_risk_profile": "UNKNOWN — models not loaded",
            "primary_concern": "Run train_models.py to generate XGBoost models",
            "recommended_option": "N/A",
            "recommendation_confidence": 0.0,
        },
        scenarios=[],
        what_if_narrative=(
            f"MOCK OUTPUT — XGBoost models not found at {MODELS_DIR}. "
            f"Run: python agents/digital_twin/train_models.py"
        ),
        fhir_care_plan=None,
        feature_attribution=[],
        models_loaded=False,
        mock=True,
    )


# ── Main Endpoint ──────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=DigitalTwinResponse)
async def simulate(request: DigitalTwinRequest) -> DigitalTwinResponse:
    """
    Run Digital Twin simulation for a patient.

    1. Feature engineering from PatientState
    2. XGBoost baseline risk prediction
    3. Treatment effect simulation for each option
    4. Scenario comparison + recommendation
    5. LLM narrative generation
    6. FHIR CarePlan assembly
    """
    if not _models_loaded:
        return mock_response(request)

    patient_state = request.patient_state
    patient_id    = patient_state.get("patient_id", "unknown")

    # Step 1: Feature engineering
    try:
        feature_vector, feature_dict = engineer_features(patient_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # Step 2: Baseline risk prediction
    try:
        baseline_risks = predict_baseline_risks(feature_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {e}")

    risk_profile = determine_patient_risk_profile(baseline_risks)

    # Step 3: Add default no-treatment baseline if not in options
    treatment_options = list(request.treatment_options)
    option_ids = {opt.option_id for opt in treatment_options}
    if "C" not in option_ids:
        treatment_options.append(TreatmentOption(
            option_id="C",
            label="No treatment (baseline)",
            drugs=[],
            interventions=[],
        ))

    # Step 4: Simulate each treatment option
    scenarios = []
    for opt in treatment_options:
        predictions = simulate_treatment(
            baseline_risks=baseline_risks,
            drugs=opt.drugs,
            interventions=opt.interventions,
        )
        # Risks for no-treatment = unmodified baseline
        if opt.option_id == "C" and not opt.drugs and not opt.interventions:
            predictions = {
                "readmission_risk_30d":   round(baseline_risks["readmission_30d"], 3),
                "mortality_risk_30d":     round(baseline_risks["mortality_30d"], 3),
                "complication_risk":      round(baseline_risks["complication"], 3),
                "recovery_probability_7d": round(
                    max(0.05, 1 - baseline_risks["mortality_30d"] - baseline_risks["complication"] * 0.3), 3
                ),
                "estimated_recovery_days": None,
            }

        key_risks = []
        if predictions["mortality_risk_30d"] > 0.10:
            key_risks.append(f"30-day mortality risk: {predictions['mortality_risk_30d']:.0%}")
        if predictions["readmission_risk_30d"] > 0.20:
            key_risks.append(f"Readmission risk: {predictions['readmission_risk_30d']:.0%}")
        if not key_risks:
            key_risks.append("Low overall risk profile with this treatment")

        scenarios.append({
            "option_id":   opt.option_id,
            "label":       opt.label,
            "drugs":       opt.drugs,
            "interventions": opt.interventions,
            "predictions": predictions,
            "key_risks":   key_risks,
        })

    # Step 5: Select recommended option (exclude C — no treatment)
    scoreable = [s for s in scenarios if s["option_id"] != "C"]
    if scoreable:
        recommended_id, rec_confidence = select_recommended_option(scoreable)
    else:
        recommended_id = scenarios[0]["option_id"] if scenarios else "A"
        rec_confidence = 0.70

    # Step 6: Feature attribution
    attribution = get_feature_attribution(feature_dict, baseline_risks)

    # Step 7: LLM narrative
    narrative = build_llm_narrative(
        patient_state=patient_state,
        diagnosis=request.diagnosis,
        scenarios=scenarios,
        recommended_option=recommended_id,
        risk_profile=risk_profile,
    )

    # Step 8: FHIR CarePlan for recommended option
    rec_option = next((opt for opt in treatment_options if opt.option_id == recommended_id), treatment_options[0])
    rec_scenario = next((s for s in scenarios if s["option_id"] == recommended_id), scenarios[0])
    rec_recovery = rec_scenario["predictions"].get("recovery_probability_7d", 0.7)

    fhir_care_plan = build_fhir_care_plan(patient_id, rec_option, narrative, rec_recovery)

    return DigitalTwinResponse(
        simulation_summary={
            "patient_risk_profile":    risk_profile,
            "baseline_risks":          {k: round(v, 3) for k, v in baseline_risks.items()},
            "primary_concern":         f"{risk_profile} risk — {request.diagnosis}",
            "recommended_option":      recommended_id,
            "recommendation_confidence": rec_confidence,
        },
        scenarios=scenarios,
        what_if_narrative=narrative,
        fhir_care_plan=fhir_care_plan,
        feature_attribution=attribution,
        models_loaded=True,
        mock=False,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "digital-twin",
        "version": "1.0.0",
        "models_loaded": _models_loaded,
        "model_error": _models_error,
        "models": list(_models.keys()),
        "llm_ready": _llm_ready,
        "features": len(FEATURE_NAMES),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006)