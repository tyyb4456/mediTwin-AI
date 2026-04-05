"""
Feature Engineering — Digital Twin Agent
Extracts ML features from PatientState dict for XGBoost inference.

Feature order MUST match training (defined in train_models.py FEATURE_NAMES).
LOINC lookups use the same codes as the Lab Analysis Agent.
"""
import json
from pathlib import Path
from typing import Optional

# Load canonical feature order from training
_FEAT_PATH = Path(__file__).parent / "models" / "feature_names.json"
if _FEAT_PATH.exists():
    with open(_FEAT_PATH) as f:
        FEATURE_NAMES: list[str] = json.load(f)
else:
    # Fallback if models haven't been trained yet
    FEATURE_NAMES = [
        "age", "gender_male", "wbc", "creatinine", "albumin", "glucose",
        "crp", "hemoglobin", "potassium", "comorbidity_count",
        "has_diabetes", "has_ckd", "has_chf", "has_copd",
        "has_atrial_fibrillation", "med_count", "critical_lab_count",
        "on_anticoagulant", "on_steroid",
    ]

# ── LOINC → feature name mapping ─────────────────────────────────────────────
LOINC_TO_FEATURE = {
    "26464-8": "wbc",         # White Blood Cell Count
    "2160-0":  "creatinine",  # Creatinine
    "1751-7":  "albumin",     # Albumin
    "2345-7":  "glucose",     # Glucose
    "1988-5":  "crp",         # C-Reactive Protein
    "718-7":   "hemoglobin",  # Hemoglobin
    "2823-3":  "potassium",   # Potassium
}

# ── ICD-10 prefix → comorbidity flag ─────────────────────────────────────────
CONDITION_FLAGS = {
    "has_diabetes":          ["E10", "E11", "E12", "E13", "E14"],
    "has_ckd":               ["N18"],
    "has_chf":               ["I50"],
    "has_copd":              ["J44", "J43"],
    "has_atrial_fibrillation": ["I48"],
}

# ── Drug name fragments → medication flags ────────────────────────────────────
ANTICOAGULANTS = {
    "warfarin", "apixaban", "rivaroxaban", "dabigatran",
    "edoxaban", "heparin", "enoxaparin",
}
STEROIDS = {
    "prednisone", "prednisolone", "methylprednisolone",
    "dexamethasone", "hydrocortisone", "budesonide",
}


def _get_lab_value(lab_results: list[dict], loinc: str) -> Optional[float]:
    """Find the first matching LOINC value from a lab results list."""
    for lab in lab_results:
        if lab.get("loinc") == loinc:
            val = lab.get("value")
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return None


def _has_condition(conditions: list[dict], icd_prefixes: list[str]) -> int:
    """Return 1 if any active condition starts with any of the given ICD-10 prefixes."""
    for cond in conditions:
        code = cond.get("code", "")
        for prefix in icd_prefixes:
            if code.startswith(prefix):
                return 1
    return 0


def _has_drug(medications: list[dict], drug_names: set[str]) -> int:
    """Return 1 if any current medication name fragment matches."""
    for med in medications:
        drug = med.get("drug", "").lower()
        for name in drug_names:
            if name in drug:
                return 1
    return 0


def _count_critical_labs(lab_results: list[dict]) -> int:
    """Count the number of lab results flagged as CRITICAL."""
    return sum(1 for lab in lab_results if lab.get("flag") == "CRITICAL")


def engineer_features(patient_state: dict) -> tuple[list[float], dict]:
    """
    Extract ML feature vector from a PatientState dict.

    Returns:
        (feature_vector, feature_dict)
        feature_vector — list of floats in FEATURE_NAMES order (for XGBoost)
        feature_dict   — human-readable dict for attribution/debugging
    """
    demographics = patient_state.get("demographics", {})
    conditions   = patient_state.get("active_conditions", [])
    medications  = patient_state.get("medications", [])
    labs         = patient_state.get("lab_results", [])

    age         = float(demographics.get("age", 50))
    gender      = demographics.get("gender", "unknown").lower()
    gender_male = 1.0 if gender in ("male", "m") else 0.0

    # Lab values — use clinical defaults when not available in PatientState
    wbc         = _get_lab_value(labs, "26464-8") or 8.0
    creatinine  = _get_lab_value(labs, "2160-0")  or 1.0
    albumin     = _get_lab_value(labs, "1751-7")  or 3.8
    glucose     = _get_lab_value(labs, "2345-7")  or 100.0
    crp         = _get_lab_value(labs, "1988-5")  or 5.0
    hemoglobin  = _get_lab_value(labs, "718-7")   or 13.0
    potassium   = _get_lab_value(labs, "2823-3")  or 4.0

    # Comorbidity flags
    has_diabetes = _has_condition(conditions, CONDITION_FLAGS["has_diabetes"])
    has_ckd      = _has_condition(conditions, CONDITION_FLAGS["has_ckd"])
    has_chf      = _has_condition(conditions, CONDITION_FLAGS["has_chf"])
    has_copd     = _has_condition(conditions, CONDITION_FLAGS["has_copd"])
    has_af       = _has_condition(conditions, CONDITION_FLAGS["has_atrial_fibrillation"])

    comorbidity_count = float(sum([has_diabetes, has_ckd, has_chf, has_copd, has_af])
                               + max(0, len(conditions) - 5))

    # Medication counts and flags
    med_count        = float(len(medications))
    on_anticoagulant = _has_drug(medications, ANTICOAGULANTS)
    on_steroid       = _has_drug(medications, STEROIDS)
    critical_lab_count = float(_count_critical_labs(labs))

    feature_dict = {
        "age":                    age,
        "gender_male":            gender_male,
        "wbc":                    wbc,
        "creatinine":             creatinine,
        "albumin":                albumin,
        "glucose":                glucose,
        "crp":                    crp,
        "hemoglobin":             hemoglobin,
        "potassium":              potassium,
        "comorbidity_count":      comorbidity_count,
        "has_diabetes":           float(has_diabetes),
        "has_ckd":                float(has_ckd),
        "has_chf":                float(has_chf),
        "has_copd":               float(has_copd),
        "has_atrial_fibrillation": float(has_af),
        "med_count":              med_count,
        "critical_lab_count":     critical_lab_count,
        "on_anticoagulant":       float(on_anticoagulant),
        "on_steroid":             float(on_steroid),
    }

    # Build vector in canonical order
    feature_vector = [feature_dict[name] for name in FEATURE_NAMES]

    return feature_vector, feature_dict