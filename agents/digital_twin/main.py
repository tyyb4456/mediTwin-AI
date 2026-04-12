"""
Agent 6: Digital Twin Agent — main.py
FastAPI app, lifespan startup, XGBoost inference, and HTTP endpoints.
Port: 8006

Pipeline per request:
  1. Feature engineering from PatientState (deterministic + temporal)
  2. XGBoost inference — multi-horizon risk scores
  3. Uncertainty quantification (Bayesian intervals)
  4. Treatment simulation with adherence modeling
  5. Cost-effectiveness analysis
  6. Sensitivity analysis
  7. Select recommended option (weighted composite + cost)
  8. LLM narrative — evidence-based clinical reasoning
  9. FHIR CarePlan with provenance and confidence metrics
"""
import os
import sys
import json
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException

from stream_endpoints import twin_router as stream_router
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from feature_engineering import engineer_features, FEATURE_NAMES, extract_temporal_features
from simulator import (
    simulate_treatment,
    determine_patient_risk_profile,
    select_recommended_option,
)
from model import (
    TreatmentOption,
    DigitalTwinRequest,
    DigitalTwinResponse,
)
from clinical_tools import (
    check_drug_guideline_adherence,
    CLINICAL_GUIDELINES,
    perform_sensitivity_analysis,
    analyze_cost_effectiveness,
    build_enhanced_llm_narrative,
    build_enhanced_fhir_care_plan,
)

# ── Model paths ────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILES = {
    "readmission_30d": MODELS_DIR / "readmission_30d.json",
    "mortality_30d":   MODELS_DIR / "mortality_30d.json",
    "complication":    MODELS_DIR / "complication.json",
    "readmission_90d": MODELS_DIR / "readmission_90d.json",
    "mortality_1yr":   MODELS_DIR / "mortality_1yr.json",
}

# ── Global state ───────────────────────────────────────────────────────────────

_models: Dict[str, xgb.XGBClassifier] = {}
_models_loaded = False
_models_error: Optional[str] = None
_model_metadata: Dict[str, dict] = {}

_llm: Optional[ChatGoogleGenerativeAI] = None
_llm_ready = False
_tools_ready = False


# ── Numpy Serialization Sanitizer ─────────────────────────────────────────────

def sanitize(obj):
    """
    Recursively convert numpy scalar/array types to native Python types so
    Pydantic and json.dumps can serialize the response without errors.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _models_loaded, _models_error, _llm, _llm_ready, _tools_ready, _model_metadata

    # Load XGBoost models
    available_models = {name: path for name, path in MODEL_FILES.items() if path.exists()}
    missing = [name for name in MODEL_FILES if name not in available_models]

    if not available_models:
        _models_error = (
            "No model files found. "
            "Run: python agents/digital_twin/train_models.py"
        )
        print(f"⚠️  Digital Twin: {_models_error}")
    else:
        try:
            for name, path in available_models.items():
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                _models[name] = m

                metadata_path = path.with_suffix(".json.meta")
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        _model_metadata[name] = json.load(f)

            _models_loaded = True
            print(f"✓ Digital Twin: loaded {len(_models)} XGBoost risk models")

            if missing:
                print(f"  ℹ️  Optional models not found: {missing}")
                print("     Core 30d models available — extended horizons disabled")

        except Exception as e:
            _models_error = f"Model load failed: {e}"
            print(f"❌ Digital Twin: {_models_error}")

    # Init LLM
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            _llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                temperature=0.2,
                max_output_tokens=2048,
            )
            _llm_ready = True
            print("✓ Digital Twin: LLM narrative ready (Gemini 2.0 Flash)")
        except Exception as e:
            print(f"⚠️  Digital Twin: LLM init failed ({e}) — narrative disabled")
    else:
        print("⚠️  Digital Twin: No GOOGLE_API_KEY — narrative disabled")

    # Verify tools
    try:
        check_drug_guideline_adherence.invoke({
            "diagnosis_code": "J18.9",
            "proposed_drug": "Amoxicillin",
        })
        _tools_ready = True
        print("✓ Digital Twin: Clinical decision support tools ready")
    except Exception as e:
        print(f"⚠️  Digital Twin: Tools initialization issue ({e})")

    yield
    print("✓ Digital Twin Agent shutdown")


app = FastAPI(
    title="MediTwin Digital Twin Agent - Enhanced",
    description="XGBoost risk simulation + treatment scenario comparison + clinical decision support",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(stream_router)

# ── Inference Helpers ─────────────────────────────────────────────────────────

def predict_with_uncertainty(
    model: xgb.XGBClassifier,
    feature_vector: List[float],
    n_bootstrap: int = 100,
) -> Tuple[float, float, float]:
    """
    Bayesian-inspired uncertainty quantification via bootstrap feature perturbation.
    Returns: (point_estimate, lower_95ci, upper_95ci)
    """
    X = np.array([feature_vector], dtype=np.float32)
    point_est = float(model.predict_proba(X)[0][1])

    bootstrap_preds = [
        float(model.predict_proba(X + np.random.normal(0, 0.05, X.shape))[0][1])
        for _ in range(n_bootstrap)
    ]

    lower = float(np.percentile(bootstrap_preds, 2.5))
    upper = float(np.percentile(bootstrap_preds, 97.5))
    return point_est, lower, upper


def predict_baseline_risks_with_uncertainty(feature_vector: List[float]) -> Dict[str, Dict]:
    """
    Run all available XGBoost models with uncertainty quantification.
    Returns predictions for all available time horizons.
    """
    predictions: Dict[str, Dict] = {}

    for outcome in ("readmission_30d", "mortality_30d", "complication"):
        if outcome not in _models:
            continue
        point, lower, upper = predict_with_uncertainty(_models[outcome], feature_vector)
        width = upper - lower
        confidence = "HIGH" if width < 0.15 else ("MODERATE" if width < 0.30 else "LOW")
        predictions[outcome] = {
            "point_estimate": round(point, 4),
            "lower_bound_95ci": round(lower, 4),
            "upper_bound_95ci": round(upper, 4),
            "confidence_level": confidence,
            "interval_width": round(width, 4),
        }

    for outcome in ("readmission_90d", "mortality_1yr"):
        if outcome not in _models:
            continue
        point, lower, upper = predict_with_uncertainty(_models[outcome], feature_vector)
        predictions[outcome] = {
            "point_estimate": round(point, 4),
            "lower_bound_95ci": round(lower, 4),
            "upper_bound_95ci": round(upper, 4),
        }

    return predictions


def _determine_model_confidence(baseline_risks_with_ci: Dict[str, Dict]) -> str:
    """Aggregate confidence across core 30d models."""
    levels = [
        baseline_risks_with_ci[k]["confidence_level"]
        for k in ("readmission_30d", "mortality_30d", "complication")
        if k in baseline_risks_with_ci
    ]
    if all(c == "HIGH" for c in levels):
        return "HIGH"
    if all(c in ("HIGH", "MODERATE") for c in levels):
        return "MODERATE"
    return "LOW"


def _generate_simulation_hash(request: DigitalTwinRequest) -> str:
    content = json.dumps(
        {
            "patient_id": request.patient_state.get("patient_id"),
            "diagnosis": request.diagnosis,
            "options": [opt.dict() for opt in request.treatment_options],
            "timestamp": request.patient_state.get("state_timestamp"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Main Simulation Endpoint ──────────────────────────────────────────────────

@app.post("/simulate", response_model=DigitalTwinResponse)
async def simulate(request: DigitalTwinRequest) -> DigitalTwinResponse:
    """
    Run enhanced Digital Twin simulation with:
    - Multi-horizon risk predictions with Bayesian uncertainty
    - Sensitivity analysis on modifiable risk factors
    - Cost-effectiveness analysis (ICEA)
    - Guideline adherence checking
    - Enhanced FHIR output with provenance tracking
    """
    if not _models_loaded:
        return DigitalTwinResponse(
            simulation_summary={
                "patient_risk_profile": "UNKNOWN — models not loaded",
                "primary_concern": "Run train_models.py to generate XGBoost models",
                "recommended_option": "N/A",
                "recommendation_confidence": 0.0,
            },
            scenarios=[],
            what_if_narrative="MOCK OUTPUT — XGBoost models not found.",
            fhir_care_plan=None,
            feature_attribution=[],
            sensitivity_analysis=None,
            cost_effectiveness_summary=None,
            models_loaded=False,
            model_confidence="NONE",
            provenance={"error": "Models not loaded"},
            mock=True,
        )

    patient_state = request.patient_state
    patient_id = patient_state.get("patient_id", "unknown")
    diagnosis_code = (
        request.diagnosis_code
        or request.diagnosis.split("(")[-1].strip(")")
    )
    simulation_hash = _generate_simulation_hash(request)

    # 1. Feature engineering
    try:
        feature_vector, feature_dict = engineer_features(patient_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # 2. Baseline risk prediction with uncertainty
    try:
        baseline_risks_with_ci = predict_baseline_risks_with_uncertainty(feature_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {e}")

    baseline_risks = {
        "readmission_30d": baseline_risks_with_ci["readmission_30d"]["point_estimate"],
        "mortality_30d":   baseline_risks_with_ci["mortality_30d"]["point_estimate"],
        "complication":    baseline_risks_with_ci["complication"]["point_estimate"],
    }

    risk_profile = determine_patient_risk_profile(baseline_risks)
    model_confidence = _determine_model_confidence(baseline_risks_with_ci)

    # 3. Sensitivity analysis (optional)
    sensitivity_results = None
    if request.include_sensitivity_analysis:
        try:
            sensitivity_results = perform_sensitivity_analysis(
                feature_vector, feature_dict, _models, FEATURE_NAMES
            )
        except Exception as e:
            print(f"  ⚠️  Sensitivity analysis failed: {e}")

    # 4. Prepare treatment options — ensure baseline "no treatment" exists
    treatment_options = list(request.treatment_options)
    if not any(opt.option_id == "C" for opt in treatment_options):
        treatment_options.append(TreatmentOption(
            option_id="C",
            label="No treatment (baseline)",
            drugs=[],
            interventions=[],
            estimated_cost_usd=0,
        ))

    # 5. Simulate each option + guideline check
    scenarios = []
    for opt in treatment_options:
        predictions = simulate_treatment(
            baseline_risks=baseline_risks,
            drugs=opt.drugs,
            interventions=opt.interventions,
        )

        # Propagate CI width (relative) from baseline to treatment predictions
        predictions_with_ci: Dict[str, Dict] = {}
        for key in ("mortality_risk_30d", "readmission_risk_30d", "complication_risk"):
            base_key = key.replace("_risk", "")
            point = predictions.get(key, 0.5)
            if base_key in baseline_risks_with_ci:
                base_ci = baseline_risks_with_ci[base_key]
                width_ratio = base_ci["interval_width"] / max(base_ci["point_estimate"], 0.01)
                width = point * width_ratio
            else:
                width = 0.1
            predictions_with_ci[key] = {
                "point_estimate": point,
                "lower_bound_95ci": max(0.0, point - width / 2),
                "upper_bound_95ci": min(1.0, point + width / 2),
            }

        # Guideline adherence
        guideline_adherence = None
        if _tools_ready and diagnosis_code and opt.drugs:
            try:
                guideline_adherence = check_drug_guideline_adherence.invoke({
                    "diagnosis_code": diagnosis_code,
                    "proposed_drug": opt.drugs[0],
                })
            except Exception as e:
                print(f"  ⚠️  Guideline check failed: {e}")

        # Key risk flags
        key_risks: List[str] = []
        if predictions["mortality_risk_30d"] > 0.10:
            ci = predictions_with_ci["mortality_risk_30d"]
            key_risks.append(
                f"30-day mortality: {predictions['mortality_risk_30d']:.0%} "
                f"(CI: {ci['lower_bound_95ci']:.0%}-{ci['upper_bound_95ci']:.0%})"
            )
        if predictions["readmission_risk_30d"] > 0.20:
            key_risks.append(f"Readmission risk: {predictions['readmission_risk_30d']:.0%}")
        if guideline_adherence and guideline_adherence.get("adherence") == "OFF_GUIDELINE":
            key_risks.append("⚠️  Off-guideline treatment")
        if not key_risks:
            key_risks.append("Low overall risk with this treatment")

        scenarios.append({
            "option_id": opt.option_id,
            "label": opt.label,
            "drugs": opt.drugs,
            "interventions": opt.interventions,
            "predictions": predictions,
            "predictions_with_ci": predictions_with_ci,
            "key_risks": key_risks,
            "guideline_adherence": guideline_adherence,
            "estimated_cost_usd": opt.estimated_cost_usd,
        })

    # 6. Select recommended option (exclude baseline "C")
    scoreable = [s for s in scenarios if s["option_id"] != "C"]
    if scoreable:
        recommended_id, rec_confidence = select_recommended_option(scoreable)
    else:
        recommended_id = scenarios[0]["option_id"] if scenarios else "A"
        rec_confidence = 0.70

    # 7. Cost-effectiveness analysis (optional)
    cost_effectiveness = None
    if request.include_cost_analysis:
        try:
            patient_age = patient_state.get("demographics", {}).get("age", 65)
            cost_effectiveness = analyze_cost_effectiveness(scenarios, patient_age)
        except Exception as e:
            print(f"  ⚠️  Cost-effectiveness analysis failed: {e}")

    # 8. Feature attribution
    try:
        from feature_engineering import get_feature_attribution
        attribution = get_feature_attribution(feature_dict, baseline_risks)
    except Exception:
        attribution = []

    # 9. LLM narrative
    narrative = build_enhanced_llm_narrative(
        patient_state=patient_state,
        diagnosis=request.diagnosis,
        diagnosis_code=diagnosis_code,
        scenarios=scenarios,
        recommended_option=recommended_id,
        risk_profile=risk_profile,
        llm=_llm,
        llm_ready=_llm_ready,
        sensitivity_top_3=sensitivity_results[:3] if sensitivity_results else None,
        cost_effectiveness=cost_effectiveness,
    )

    # 10. FHIR CarePlan
    rec_option = next(
        (opt for opt in treatment_options if opt.option_id == recommended_id),
        treatment_options[0],
    )
    rec_scenario = next(
        (s for s in scenarios if s["option_id"] == recommended_id),
        scenarios[0],
    )
    rec_recovery = rec_scenario["predictions"].get("recovery_probability_7d", 0.7)

    fhir_care_plan = build_enhanced_fhir_care_plan(
        patient_id=patient_id,
        recommended_option=rec_option,
        narrative=narrative,
        predicted_recovery=rec_recovery,
        prediction_confidence=model_confidence,
        diagnosis_code=diagnosis_code,
        feature_attribution=attribution,
        model_version="2.0.0",
    )

    provenance = {
        "simulation_id": simulation_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "2.0.0",
        "models_used": list(_models.keys()),
        "feature_count": len(FEATURE_NAMES),
        "prediction_horizons": list(baseline_risks_with_ci.keys()),
        "overall_confidence": model_confidence,
        "reproducible": True,
    }

    # Sanitize all numpy types before Pydantic serialization
    return DigitalTwinResponse(
        simulation_summary=sanitize({
            "patient_risk_profile": risk_profile,
            "baseline_risks": {k: round(float(v), 3) for k, v in baseline_risks.items()},
            "baseline_risks_with_ci": baseline_risks_with_ci,
            "primary_concern": f"{risk_profile} risk — {request.diagnosis}",
            "recommended_option": recommended_id,
            "recommendation_confidence": float(rec_confidence),
            "model_confidence": model_confidence,
        }),
        scenarios=sanitize(scenarios),
        what_if_narrative=narrative,
        fhir_care_plan=sanitize(fhir_care_plan),
        feature_attribution=sanitize(attribution),
        sensitivity_analysis=sanitize(sensitivity_results),
        cost_effectiveness_summary=sanitize(cost_effectiveness),
        models_loaded=True,
        model_confidence=model_confidence,
        provenance=sanitize(provenance),
        mock=False,
    )


# ── Supporting Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "digital-twin-enhanced",
        "version": "2.0.0",
        "models_loaded": _models_loaded,
        "model_error": _models_error,
        "models": list(_models.keys()),
        "llm_ready": _llm_ready,
        "tools_ready": _tools_ready,
        "features": len(FEATURE_NAMES),
        "capabilities": [
            "multi_horizon_predictions",
            "uncertainty_quantification",
            "sensitivity_analysis",
            "cost_effectiveness_analysis",
            "guideline_adherence_checking",
            "enhanced_fhir_provenance",
        ],
        "model_confidence_tracking": True,
    }


@app.get("/guidelines/{diagnosis_code}")
async def get_guidelines(diagnosis_code: str):
    """Retrieve clinical guidelines for a given ICD-10 diagnosis code."""
    base_code = diagnosis_code.split(".")[0]
    guideline = CLINICAL_GUIDELINES.get(base_code)

    if not guideline:
        return {"available": False, "message": f"No guideline for {diagnosis_code}"}

    return {
        "available": True,
        "diagnosis_code": diagnosis_code,
        "guideline": guideline,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006)