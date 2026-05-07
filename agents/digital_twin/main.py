"""
Agent 6: Digital Twin Agent — main.py (ENHANCED - P0/P1 FIXES)
FastAPI app, lifespan startup, XGBoost inference, and HTTP endpoints.
Port: 8006

P0 FIXES APPLIED:
  #1 - 90d/1yr model loading verified (requires fixed train_models.py)
  #2 - Pharmacokinetic dose adjustments added to scenarios

P1 FIXES APPLIED:
  #1 - CURB-65 / Charlson scores surfaced in simulation_summary
  #2 - Drug combination synergy effects explained in scenarios
"""


import logging
logger = logging.getLogger("digital_twin_agent")

try:
    import db
    from db import save_simulation, SimulationRecord
    _db_available = True
except ImportError:
    _db_available = False
    logger.error("   ✘   Database module not available - persistence disabled")

from temporal_effects import (
    predict_temporal_trajectory,
    get_treatment_profile_key,
    add_temporal_effects_to_scenario,
)


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
    COMBINATION_EFFECTS,  # ← P1 FIX: import for synergy detection
    estimate_drug_half_life_adjustment,  # ← P0 FIX #2: import PK function
)
from model import (
    TreatmentOption,
    DigitalTwinRequest,
    DigitalTwinResponse,
)
from clinical_tools import (
    check_drug_guideline_adherence,
    check_allergy_contraindications,
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

async def _save_simulation_to_db(
    request_id: str,
    patient_state: dict,
    request: "DigitalTwinRequest",
    response: "DigitalTwinResponse",
    elapsed_ms: int,
) -> tuple[bool, Optional[int]]:
    """
    Save simulation to database and return (success, row_id).
    Non-blocking - failures are logged but don't break the API response.
    """
    if not _db_available:
        return False, None
    
    try:
        record = SimulationRecord(
            request_id=request_id,
            patient_id=patient_state.get("patient_id", "unknown"),
            diagnosis=request.diagnosis,
            diagnosis_code=request.diagnosis_code,
            patient_risk_profile=response.simulation_summary["patient_risk_profile"],
            baseline_mortality_30d=response.simulation_summary["baseline_risks"]["mortality_30d"],
            baseline_readmission_30d=response.simulation_summary["baseline_risks"]["readmission_30d"],
            baseline_complication=response.simulation_summary["baseline_risks"]["complication"],
            recommended_option=response.simulation_summary["recommended_option"],
            recommendation_confidence=response.simulation_summary["recommendation_confidence"],
            model_confidence=response.model_confidence,
            treatment_options_count=len(request.treatment_options),
            scenarios=response.scenarios,
            simulation_summary=response.simulation_summary,
            what_if_narrative=response.what_if_narrative,
            fhir_care_plan=response.fhir_care_plan,
            feature_attribution=response.feature_attribution,
            sensitivity_analysis=response.sensitivity_analysis,
            cost_effectiveness=response.cost_effectiveness_summary,
            models_loaded=response.models_loaded,
            cache_hit=False,
            elapsed_ms=elapsed_ms,
            source="simulate",
        )
        
        row_id = await save_simulation(record)
        return (row_id is not None), row_id
    
    except Exception as e:
        logger.error(f"  ✘   Database save failed (non-fatal): {e}")
        return False, None
 

# ── P1 FIX: Clinical Risk Score Calculator ───────────────────────────────────

def calculate_clinical_risk_scores(patient_state: dict, feature_dict: dict) -> dict:
    """
    P1 FIX: Calculate CURB-65 and Charlson scores from patient state.
    Returns dict with scores, interpretations, and clinical recommendations.
    """
    from feature_engineering import _calculate_curb65_score, _calculate_charlson_index
    
    demographics = patient_state.get("demographics", {})
    conditions = patient_state.get("active_conditions", [])
    
    age = demographics.get("age", 65)
    wbc = feature_dict.get("wbc", 8.0)
    creatinine = feature_dict.get("creatinine", 1.0)
    
    curb65_score = _calculate_curb65_score(
        age=int(age),
        wbc=wbc,
        creatinine=creatinine,
    )
    
    charlson_index = _calculate_charlson_index(
        conditions=conditions,
        age=int(age),
    )
    
    # CURB-65 interpretation
    if curb65_score == 0:
        curb65_interpretation = "LOW RISK — outpatient management appropriate"
        curb65_mortality = "0.6%"
    elif curb65_score == 1:
        curb65_interpretation = "LOW RISK — outpatient or short observation"
        curb65_mortality = "2.7%"
    elif curb65_score == 2:
        curb65_interpretation = "MODERATE RISK — consider hospital admission"
        curb65_mortality = "6.8%"
    elif curb65_score == 3:
        curb65_interpretation = "HIGH RISK — hospitalization recommended"
        curb65_mortality = "14.0%"
    else:  # 4-5
        curb65_interpretation = "SEVERE RISK — ICU consideration, possible sepsis"
        curb65_mortality = "27.8% (score 4) or 50%+ (score 5)"
    
    # Charlson interpretation
    if charlson_index <= 1:
        charlson_interpretation = "LOW comorbidity burden — 10-year mortality ~10%"
    elif charlson_index <= 3:
        charlson_interpretation = "MODERATE comorbidity burden — 10-year mortality ~25-50%"
    elif charlson_index <= 5:
        charlson_interpretation = "HIGH comorbidity burden — 10-year mortality ~50-75%"
    else:
        charlson_interpretation = "SEVERE comorbidity burden — 10-year mortality >75%"
    
    return {
        "curb65": {
            "score": curb65_score,
            "interpretation": curb65_interpretation,
            "30d_mortality_estimate": curb65_mortality,
        },
        "charlson": {
            "score": charlson_index,
            "interpretation": charlson_interpretation,
        },
    }


# ── P0 FIX #2: Pharmacokinetic Dose Adjustment Calculator ─────────────────────

def calculate_pk_adjustments(
    drugs: List[str],
    patient_state: dict,
    feature_dict: dict,
) -> List[Dict]:
    """
    P0 FIX #2: Calculate pharmacokinetic dose adjustments for renally/hepatically cleared drugs.
    Returns list of adjustment recommendations with rationales.
    """
    adjustments = []
    
    demographics = patient_state.get("demographics", {})
    age = demographics.get("age", 65)
    creatinine = feature_dict.get("creatinine", 1.0)
    
    # Estimate eGFR using simplified Cockcroft-Gault
    # eGFR ≈ (140 - age) × weight / (72 × Cr)
    # Assume average weight 70kg
    egfr_estimate = (140 - age) * 70 / (72 * max(creatinine, 0.6))
    
    # CKD staging
    if egfr_estimate >= 60:
        ckd_stage = "Normal/Stage 1-2"
        renal_function = "normal"
    elif egfr_estimate >= 45:
        ckd_stage = "Stage 3a (mild-moderate)"
        renal_function = "mildly_impaired"
    elif egfr_estimate >= 30:
        ckd_stage = "Stage 3b (moderate-severe)"
        renal_function = "moderately_impaired"
    elif egfr_estimate >= 15:
        ckd_stage = "Stage 4 (severe)"
        renal_function = "severely_impaired"
    else:
        ckd_stage = "Stage 5 (end-stage)"
        renal_function = "end_stage"
    
    # Check for hepatic impairment (albumin as proxy)
    albumin = feature_dict.get("albumin", 3.8)
    hepatic_impaired = albumin < 3.0

    current_meds = patient_state.get("medications", [])
    for med in current_meds:
        med_name = med.get("drug", "")
        
        # Check if current med interacts with proposed drugs
        if "warfarin" in med_name.lower():
            coprescribed_with_azithro = any("azithromycin" in d.lower() for d in drugs)
            coprescribed_with_fluoroquinolone = any(
                fq in d.lower() for fq in ["levofloxacin", "moxifloxacin"] for d in drugs
            )
            
            if coprescribed_with_azithro:
                adjustments.append({
                    "drug": med_name,
                    "standard_dose": "5mg daily (individualized to INR)",
                    "adjusted_dose": "Reduce by 20-30% empirically OR hold 1-2 doses",
                    "rationale": "Azithromycin CYP3A4 inhibition increases warfarin effect — expect 20-40% INR rise",
                    "monitoring": "Check baseline INR, recheck at 48-72h, target INR 2.0-3.0 for AF",
                })
    
    # Drug-specific adjustments
    for drug in drugs:
        drug_lower = drug.lower()
        adjustment = None
        
        # Levofloxacin / Moxifloxacin (fluoroquinolones - renally cleared)
        if "levofloxacin" in drug_lower or "moxifloxacin" in drug_lower:
            if renal_function == "normal":
                adjustment = {
                    "drug": drug,
                    "standard_dose": "750mg once daily (levofloxacin) or 400mg once daily (moxifloxacin)",
                    "adjusted_dose": "No adjustment needed",
                    "rationale": f"Normal renal function (eGFR ~{egfr_estimate:.0f} mL/min)",
                }
            elif renal_function == "mildly_impaired":
                adjustment = {
                    "drug": drug,
                    "standard_dose": "750mg once daily",
                    "adjusted_dose": "750mg loading dose, then 500mg daily",
                    "rationale": f"CKD {ckd_stage} — 33% dose reduction recommended",
                }
            else:
                adjustment = {
                    "drug": drug,
                    "standard_dose": "750mg once daily",
                    "adjusted_dose": "500mg loading dose, then 250mg daily",
                    "rationale": f"CKD {ckd_stage} — 50% dose reduction required",
                }
        
        # Ceftriaxone (minimal renal adjustment but interaction with calcium)
        elif "ceftriaxone" in drug_lower:
            adjustment = {
                "drug": drug,
                "standard_dose": "1-2g IV once daily",
                "adjusted_dose": "No adjustment needed" if egfr_estimate > 30 else "Monitor closely",
                "rationale": f"Ceftriaxone safe in mild-moderate renal impairment (eGFR ~{egfr_estimate:.0f})",
                "warning": "Avoid calcium-containing IV solutions (risk of precipitation)" if "iv" in drug_lower else None,
            }
        
        # Azithromycin (hepatically cleared - caution in liver disease)
        elif "azithromycin" in drug_lower:
            if hepatic_impaired:
                adjustment = {
                    "drug": drug,
                    "standard_dose": "500mg daily",
                    "adjusted_dose": "Use with caution — consider 3-day course instead of 5-day",
                    "rationale": f"Hepatic impairment suspected (albumin {albumin:.1f} g/dL) — azithromycin hepatically metabolized",
                    "monitoring": "Monitor LFTs, watch for cholestatic jaundice",
                }
            else:
                adjustment = {
                    "drug": drug,
                    "standard_dose": "500mg daily",
                    "adjusted_dose": "No adjustment needed",
                    "rationale": f"Normal hepatic function (albumin {albumin:.1f} g/dL), no renal adjustment required",
                }
        
        # Warfarin (monitor INR with any antibiotic)
        elif "warfarin" in drug_lower:
            # Check if co-prescribed with azithromycin
            coprescribed_with_azithro = any("azithromycin" in d.lower() for d in drugs)
            coprescribed_with_fluoroquinolone = any(
                fq in d.lower() for fq in ["levofloxacin", "moxifloxacin"] for d in drugs
            )
            
            if coprescribed_with_azithro:
                adjustment = {
                    "drug": drug,
                    "standard_dose": "5mg daily (individualized to INR)",
                    "adjusted_dose": "Reduce by 20-30% empirically OR hold 1-2 doses",
                    "rationale": "Azithromycin CYP3A4 inhibition increases warfarin effect — expect 20-40% INR rise",
                    "monitoring": "Check baseline INR, recheck at 48-72h, target INR 2.0-3.0 for AF",
                }
            elif coprescribed_with_fluoroquinolone:
                adjustment = {
                    "drug": drug,
                    "standard_dose": "5mg daily",
                    "adjusted_dose": "Reduce by 10-20% OR monitor INR closely",
                    "rationale": "Fluoroquinolones may potentiate warfarin anticoagulation",
                    "monitoring": "Recheck INR at 3-5 days",
                }
        
        if adjustment:
            adjustments.append(adjustment)
    
    return adjustments


# ── P1 FIX: Combination Synergy Detector ──────────────────────────────────────

def detect_combination_synergy(drugs: List[str]) -> Optional[Dict]:
    """
    P1 FIX: Detect drug combination synergy effects and explain the mechanism.
    Returns synergy details or None if no synergy detected.
    
    BUGFIX: Corrected index tracking in pairwise loop to properly reference original drug names.
    """
    drug_keys = [drug.lower().split()[0].rstrip(".,") for drug in drugs]
    
    # Normalize combination drug names
    for i, key in enumerate(drug_keys):
        if "clavulanate" in drugs[i].lower() and "amoxicillin" in drugs[i].lower():
            drug_keys[i] = "amoxicillin-clavulanate"
    
    # Check all pairwise combinations
    for i, drug_a in enumerate(drug_keys):
        for j, drug_b in enumerate(drug_keys[i+1:], start=i+1):  # Track actual index
            pair = tuple(sorted([drug_a, drug_b]))
            combo_effect = COMBINATION_EFFECTS.get(pair)
            
            if combo_effect:
                # Found synergy
                return {
                    "synergy_detected": True,
                    "drug_combination": f"{drugs[i]} + {drugs[j]}",  # Use correct indices
                    "mechanism": _get_synergy_mechanism(drug_a, drug_b),
                    "additional_mortality_reduction": combo_effect.get("mortality_30d", 0),
                    "additional_complication_reduction": combo_effect.get("complication", 0),
                    "additional_recovery_days": combo_effect.get("recovery_days", 0),
                    "evidence_level": _get_evidence_level(drug_a, drug_b),
                }
    
    return None


def _get_synergy_mechanism(drug_a: str, drug_b: str) -> str:
    """Return clinical explanation of synergy mechanism."""
    pair = tuple(sorted([drug_a, drug_b]))
    
    mechanisms = {
        ("azithromycin", "ceftriaxone"): "Beta-lactam + macrolide dual coverage — ceftriaxone targets cell wall synthesis, azithromycin inhibits protein synthesis. Guideline-recommended combination for severe CAP.",
        ("azithromycin", "amoxicillin-clavulanate"): "Beta-lactam/beta-lactamase inhibitor + macrolide — broader spectrum coverage with atypical pathogen activity.",
        ("azithromycin", "prednisone"): "Antibiotic + corticosteroid — steroid reduces inflammatory response, accelerates clinical resolution but may slightly increase infection complications.",
        ("iv fluids", "*"): "Supportive care synergy — adequate hydration improves antibiotic distribution and renal clearance of toxins.",
    }
    
    return mechanisms.get(pair, f"Synergistic effect between {drug_a} and {drug_b} from combination therapy")


def _get_evidence_level(drug_a: str, drug_b: str) -> str:
    """Return evidence strength for combination."""
    pair = tuple(sorted([drug_a, drug_b]))
    
    if pair in [("azithromycin", "ceftriaxone"), ("azithromycin", "amoxicillin-clavulanate")]:
        return "1A (guideline-recommended combination for CAP)"
    elif pair == ("azithromycin", "prednisone"):
        return "1B (RCT evidence for severe CAP)"
    else:
        return "2C (observational evidence)"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _models_loaded, _models_error, _llm, _llm_ready, _tools_ready, _model_metadata

    available_models = {name: path for name, path in MODEL_FILES.items() if path.exists()}
    missing = [name for name in MODEL_FILES if name not in available_models]

    if not available_models:
        _models_error = (
            "No model files found. "
            "Run: python agents/digital_twin/train_models.py"
        )
        logger.error(f"  ✘   Digital Twin: {_models_error}")
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
            logger.info(f"   ✔   Digital Twin: loaded {len(_models)} XGBoost risk models")
            
            # P0 FIX #1: Explicit notification if extended models are missing
            if "readmission_90d" in missing or "mortality_1yr" in missing:
                logger.warning(f"  ⚠   Extended horizon models not found: {missing}")
                logger.info(f"     Run train_models.py to enable 90d/1yr predictions")

        except Exception as e:
            _models_error = f"Model load failed: {e}"
            logger.error(f"  ✘   Digital Twin: {_models_error}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            _llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                temperature=0.2,
                max_output_tokens=2048,
            )
            _llm_ready = True
            logger.info("   ✔   Digital Twin: LLM narrative ready (Gemini 2.5 Flash lite)")
        except Exception as e:
            logger.error(f"  ✘   Digital Twin: LLM init failed ({e}) — narrative disabled")
    else:
        logger.warning("  ⚠   Digital Twin: No GOOGLE_API_KEY — narrative disabled")

    try:
        check_drug_guideline_adherence.invoke({
            "diagnosis_code": "J18.9",
            "proposed_drug": "Amoxicillin",
        })
        _tools_ready = True
        logger.info("   ✔   Digital Twin: Clinical decision support tools ready")
    except Exception as e:
        logger.error(f"  ✘   Digital Twin: Tools initialization issue ({e})")

    await db.init()

    yield
    logger.info("   ✔   Digital Twin Agent shutdown")


app = FastAPI(
    title="MediTwin Digital Twin Agent - Complete (P0/P1/P2/P3)",
    description="XGBoost risk simulation + temporal treatment trajectories + database persistence",
    version="2.2.0",
    lifespan=lifespan,
)

app.include_router(stream_router)

from history_router import router as history_router
app.include_router(history_router, prefix="/history", tags=["history"])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Inference Helpers ─────────────────────────────────────────────────────────

def predict_with_uncertainty(
    model: xgb.XGBClassifier,
    feature_vector: List[float],
    n_bootstrap: int = 200,
) -> Tuple[float, float, float]:
    """
    Uncertainty quantification via bootstrap feature perturbation.
    Fixed symmetric CI calculation (Bug #2 from original fixes).
    """
    X = np.array([feature_vector], dtype=np.float32)
    point_est = float(model.predict_proba(X)[0][1])

    fv = np.array(feature_vector, dtype=np.float32)
    noise_scale = np.maximum(np.abs(fv) * 0.05, 0.01)

    rng = np.random.default_rng(42)
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        noise = rng.normal(0, noise_scale, fv.shape).astype(np.float32)
        X_perturbed = (fv + noise).reshape(1, -1)
        bootstrap_preds.append(float(model.predict_proba(X_perturbed)[0][1]))

    raw_lower = float(np.percentile(bootstrap_preds, 2.5))
    raw_upper = float(np.percentile(bootstrap_preds, 97.5))

    # Symmetric half-width CI
    half_width = (raw_upper - raw_lower) / 2.0
    lower = max(0.0, point_est - half_width)
    upper = min(1.0, point_est + half_width)

    return point_est, lower, upper


def predict_baseline_risks_with_uncertainty(feature_vector: List[float]) -> Dict[str, Dict]:
    predictions: Dict[str, Dict] = {}

    for outcome in ("readmission_30d", "mortality_30d", "complication"):
        if outcome not in _models:
            continue
        point, lower, upper = predict_with_uncertainty(_models[outcome], feature_vector)
        width = upper - lower
        confidence = "HIGH" if width < 0.15 else ("MODERATE" if width < 0.30 else "LOW")
        predictions[outcome] = {
            "point_estimate":    round(point, 4),
            "lower_bound_95ci":  round(lower, 4),
            "upper_bound_95ci":  round(upper, 4),
            "confidence_level":  confidence,
            "interval_width":    round(width, 4),
        }

    for outcome in ("readmission_90d", "mortality_1yr"):
        if outcome not in _models:
            continue
        point, lower, upper = predict_with_uncertainty(_models[outcome], feature_vector)
        predictions[outcome] = {
            "point_estimate":   round(point, 4),
            "lower_bound_95ci": round(lower, 4),
            "upper_bound_95ci": round(upper, 4),
        }

    return predictions


def _determine_model_confidence(baseline_risks_with_ci: Dict[str, Dict]) -> str:
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

def _resolve_option_cost(opt) -> tuple:
    """Returns (cost_usd: float, cost_source: str)."""
    from simulator import DRUG_EFFECTS
 
    raw = opt.estimated_cost_usd
 
    if raw is not None and raw > 0:
        return float(raw), "provided"
 
    if opt.option_id == "C" or (not opt.drugs and not opt.interventions):
        return 0.0, "zero"
 
    imputed = 0.0
    for drug in opt.drugs:
        key = drug.lower().split()[0].rstrip(".,")
        if "clavulanate" in drug.lower():
            key = "amoxicillin-clavulanate"
        imputed += DRUG_EFFECTS.get(key, {}).get("cost_usd", 50)
 
    for iv in opt.interventions:
        iv_lower = iv.lower()
        if "hospitalization" in iv_lower:
            imputed += 12_000
        elif "iv fluids" in iv_lower:
            imputed += 250
        elif "monitoring" in iv_lower:
            imputed += 2_000
        elif "o2" in iv_lower or "oxygen" in iv_lower:
            imputed += 150
        else:
            imputed += 100
 
    imputed += 200
    return round(imputed, 2), "imputed"

# ── Main Simulation Endpoint ──────────────────────────────────────────────────

@app.post("/simulate", response_model=DigitalTwinResponse)
async def simulate(request: DigitalTwinRequest) -> DigitalTwinResponse:

    from shared.sse_utils import Timer
    simulation_timer = Timer()


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
    patient_id    = patient_state.get("patient_id", "unknown")
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

    risk_profile      = determine_patient_risk_profile(baseline_risks)
    model_confidence  = _determine_model_confidence(baseline_risks_with_ci)
    
    # P1 FIX #1: Calculate clinical risk scores
    clinical_risk_scores = calculate_clinical_risk_scores(patient_state, feature_dict)

    # 3. Sensitivity analysis (optional)
    sensitivity_results = None
    if request.include_sensitivity_analysis:
        try:
            sensitivity_results = perform_sensitivity_analysis(
                feature_vector, feature_dict, _models, FEATURE_NAMES
            )
        except Exception as e:
            logger.warning(f"  ⚠  Sensitivity analysis failed: {e}")

    # 4. Prepare treatment options
    treatment_options = list(request.treatment_options)
    if not any(opt.option_id == "C" for opt in treatment_options):
        treatment_options.append(TreatmentOption(
            option_id="C",
            label="No treatment (baseline)",
            drugs=[],
            interventions=[],
            estimated_cost_usd=0,
        ))

    # 5. Simulate each option + enhancements
    scenarios = []
    for opt in treatment_options:
        predictions = simulate_treatment(
            baseline_risks=baseline_risks,
            drugs=opt.drugs,
            interventions=opt.interventions,
        )

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
                "point_estimate":    round(point, 4),
                "lower_bound_95ci":  round(max(0.0, point - width / 2), 4),
                "upper_bound_95ci":  round(min(1.0, point + width / 2), 4),
            }

        # Guideline adherence
        guideline_adherence = None
        if _tools_ready and diagnosis_code and opt.drugs:
            try:
                adherence_results = []
                for drug in opt.drugs:
                    result = check_drug_guideline_adherence.invoke({
                        "diagnosis_code": diagnosis_code,
                        "proposed_drug": drug,
                    })
                    adherence_results.append({**result, "drug": drug})
                priority = {
                    "OFF_GUIDELINE": 0, "UNKNOWN": 1, "SECOND_LINE": 2,
                    "INPATIENT_APPROPRIATE": 3, "GUIDELINE_LISTED": 3, "FIRST_LINE": 4,
                }
                adherence_results.sort(key=lambda r: priority.get(r.get("adherence", "UNKNOWN"), 1))
                guideline_adherence = adherence_results
            except Exception as e:
                logger.warning(f"  ⚠  Guideline check failed: {e}")

        # Safety check
        safety_check = None
        if _tools_ready:
            try:
                allergies     = patient_state.get("allergies", [])
                current_meds  = patient_state.get("medications", [])
                safety_check  = check_allergy_contraindications.invoke({
                    "proposed_drugs":        opt.drugs + opt.interventions,
                    "allergies":             allergies,
                    "current_medications":   current_meds,
                })
            except Exception as e:
                logger.warning(f"  ⚠  Safety check failed: {e}")
        
        # P0 FIX #2: Pharmacokinetic dose adjustments
        pk_adjustments = calculate_pk_adjustments(opt.drugs, patient_state, feature_dict)
        
        # P1 FIX #2: Detect combination synergy
        synergy = detect_combination_synergy(opt.drugs) if len(opt.drugs) >= 2 else None



        # Key risks
        key_risks: List[str] = []

        if safety_check:
            for alert in safety_check.get("allergy_alerts", []):
                key_risks.append(f" ☢ {alert['alert']}")
            for interaction in safety_check.get("interaction_alerts", []):
                key_risks.append(
                    f" ⚠  DDI: {interaction['warning']} "
                    f"({interaction['proposed_drug']} ↔ {interaction['existing_drug']})"
                )

        if predictions["mortality_risk_30d"] > 0.05:
            ci = predictions_with_ci["mortality_risk_30d"]
            key_risks.append(
                f"30-day mortality: {predictions['mortality_risk_30d']:.0%} "
                f"(CI: {ci['lower_bound_95ci']:.0%}-{ci['upper_bound_95ci']:.0%})"
            )
        if predictions["readmission_risk_30d"] > 0.15:
            key_risks.append(f"Readmission risk: {predictions['readmission_risk_30d']:.0%}")

        if guideline_adherence:
            for g in (guideline_adherence if isinstance(guideline_adherence, list) else [guideline_adherence]):
                if g.get("adherence") == "OFF_GUIDELINE":
                    key_risks.append(
                        f"  ⚠   Off-guideline: {g.get('drug', '')} — {g.get('message', '')}"
                    )

        if not key_risks:
            key_risks.append("Low overall risk with this treatment")

        resolved_cost, cost_source = _resolve_option_cost(opt)
 
        # Add temporal treatment trajectory
        demographics = patient_state.get("demographics", {})
        patient_age = demographics.get("age", 65)
        comorbidity_count = int(feature_dict.get("comorbidity_count", 0))
        critical_lab_count = int(feature_dict.get("critical_lab_count", 0))
        
        scenario_data = {
            "option_id":           opt.option_id,
            "label":               opt.label,
            "drugs":               opt.drugs,
            "interventions":       opt.interventions,
            "predictions":         predictions,
            "predictions_with_ci": predictions_with_ci,
            "key_risks":           key_risks,
            "guideline_adherence": guideline_adherence,
            "safety_check":        safety_check,
            "estimated_cost_usd":  resolved_cost,
            "cost_source":         cost_source,
            "pharmacokinetic_adjustments": pk_adjustments,
            "combination_synergy": synergy,
        }
        
        # P3 FIX: Add temporal effects if not baseline option
        if opt.option_id != "C":
            scenario_data = add_temporal_effects_to_scenario(
                scenario_data,
                baseline_risks,
                patient_age,
                comorbidity_count,
                critical_lab_count,
            )
        
        scenarios.append(scenario_data)

    # 6. Select recommended option
    patient_prefs        = request.patient_preferences or {}
    prioritize_cost      = patient_prefs.get("prioritize_cost", False)
    avoid_hospitalization = patient_prefs.get("avoid_hospitalization", False)

    scoreable = [s for s in scenarios if s["option_id"] != "C"]

    safe_scoreable = [
        s for s in scoreable
        if (s.get("safety_check") or {}).get("safety_flag") != "CONTRAINDICATED"
    ]
    if not safe_scoreable:
        safe_scoreable = scoreable
        logger.warning("  ⚠   All treatment options flagged CONTRAINDICATED")

    if avoid_hospitalization:
        non_hosp = [
            s for s in safe_scoreable
            if "hospitalization" not in [i.lower() for i in s.get("interventions", [])]
        ]
        if non_hosp:
            safe_scoreable = non_hosp

    if safe_scoreable:
        recommended_id, rec_confidence = select_recommended_option(
            safe_scoreable, prioritize_cost=prioritize_cost
        )
    else:
        recommended_id  = scenarios[0]["option_id"] if scenarios else "A"
        rec_confidence  = 0.70

    # 7. Cost-effectiveness analysis
    cost_effectiveness = None
    if request.include_cost_analysis:
        try:
            patient_age = patient_state.get("demographics", {}).get("age", 65)
            cost_effectiveness = analyze_cost_effectiveness(scenarios, patient_age)
        except Exception as e:
            logger.warning(f"  ⚠   Cost-effectiveness analysis failed: {e}")

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
        model_version="2.1.0",
    )

    # Provenance with horizon tracking
    requested_horizons = request.prediction_horizons or ["7d", "30d"]
    available_horizons = list(baseline_risks_with_ci.keys())

    has_7d = any(
        s.get("predictions", {}).get("recovery_probability_7d") is not None
        for s in scenarios
    )

    horizon_map = {
        "7d":  lambda: has_7d,
        "30d": lambda: any(k in available_horizons for k in ["readmission_30d", "mortality_30d", "complication"]),
        "90d": lambda: "readmission_90d" in available_horizons,
        "1yr": lambda: "mortality_1yr" in available_horizons,
    }

    fulfilled_horizons = [
        h for h in requested_horizons
        if horizon_map.get(h, lambda: False)()
    ]

    elapsed_ms = simulation_timer.elapsed_ms()
    
    # Build initial provenance
    provenance = {
        "simulation_id":        simulation_hash,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "model_version":        "2.1.0",
        "models_used":          list(_models.keys()),
        "feature_count":        len(FEATURE_NAMES),
        "requested_horizons":   requested_horizons,
        "fulfilled_horizons":   fulfilled_horizons,
        "unfulfilled_horizons": [h for h in requested_horizons if h not in fulfilled_horizons],
        "horizon_gap_reason": (
            "Extended models (readmission_90d, mortality_1yr) not loaded — "
            "run train_models.py to enable 90d and 1yr predictions."
            if any(h not in fulfilled_horizons for h in requested_horizons)
            else None
        ),
        "available_model_keys": available_horizons,
        "overall_confidence":   model_confidence,
        "reproducible":         True,
        "enhancements_applied": [
            "pharmacokinetic_dose_adjustments",
            "clinical_risk_scores",
            "combination_synergy_detection",
        ],
        "elapsed_ms": elapsed_ms,
    }

    # Build response first
    response = DigitalTwinResponse(
        simulation_summary=sanitize({
            "patient_risk_profile":       risk_profile,
            "baseline_risks":             {k: round(float(v), 3) for k, v in baseline_risks.items()},
            "baseline_risks_with_ci":     baseline_risks_with_ci,
            "clinical_risk_scores":       clinical_risk_scores,  # ← P1 FIX #1
            "primary_concern":            f"{risk_profile} risk — {request.diagnosis}",
            "recommended_option":         recommended_id,
            "recommendation_confidence":  float(rec_confidence),
            "model_confidence":           model_confidence,
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

    db_saved, row_id = await _save_simulation_to_db(
        simulation_hash, patient_state, request, response, elapsed_ms
    )
    
    response.provenance["database"] = {
        "saved": db_saved,
        "row_id": row_id,
        "retrieval_endpoint": f"/history/request/{simulation_hash}" if db_saved else None,
        "available": _db_available,
    }
    
    return response

# ── Supporting Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":           "healthy",
        "agent":            "digital-twin-enhanced-p0-p1-p2-p3",
        "version":          "2.2.0",  # ← Updated
        "models_loaded":    _models_loaded,
        "model_error":      _models_error,
        "models":           list(_models.keys()),
        "llm_ready":        _llm_ready,
        "tools_ready":      _tools_ready,
        "features":         len(FEATURE_NAMES),
        "capabilities": [
            "multi_horizon_predictions",
            "uncertainty_quantification",
            "sensitivity_analysis",
            "cost_effectiveness_analysis",
            "guideline_adherence_checking",
            "enhanced_fhir_provenance",
            "pharmacokinetic_dose_adjustments",
            "clinical_risk_scores",
            "combination_synergy_detection",
            "temporal_treatment_effects",      # ← P3 FIX
            "database_persistence",             # ← P2 FIX
        ],
        "model_confidence_tracking": True,
        "p0_p1_fixes_applied": True,
        "p2_p3_fixes_applied": True,  # ← New flag
        "database_available": _db_available,
    }


@app.get("/guidelines/{diagnosis_code}")
async def get_guidelines(diagnosis_code: str):
    base_code = diagnosis_code.split(".")[0]
    guideline = CLINICAL_GUIDELINES.get(base_code)

    if not guideline:
        return {"available": False, "message": f"No guideline for {diagnosis_code}"}

    return {
        "available":        True,
        "diagnosis_code":   diagnosis_code,
        "guideline":        guideline,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)