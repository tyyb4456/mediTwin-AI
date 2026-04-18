"""
clinical_tools.py — Clinical guidelines, LangChain decision support tools,
                    and analysis functions (sensitivity, cost-effectiveness,
                    narrative generation, FHIR CarePlan builder).

FIXES APPLIED:
  #1  - Guideline matching: fixed string containment direction (proposed ⊃ guideline, not vice-versa)
  #4  - CEA: exclude no-treatment baseline (Option C) from most_cost_effective
  #6  - Sensitivity narrative: filter non-modifiable features before passing to LLM
  #7  - Float precision: round CI bounds to 4 dp throughout
  #8  - Sensitivity sort: modifiable features ranked first, then by magnitude
"""
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from model import TreatmentOption

# ── Clinical Guidelines Database ──────────────────────────────────────────────

CLINICAL_GUIDELINES: Dict[str, dict] = {
    "J18.9": {  # Pneumonia
        "first_line": ["Amoxicillin-clavulanate", "Azithromycin"],
        "second_line": ["Levofloxacin", "Moxifloxacin"],
        "inpatient": ["Ceftriaxone IV + Azithromycin", "Ceftriaxone"],
        "duration_days": {"outpatient": 5, "inpatient": 7},
        "monitoring": ["WBC", "CRP", "Chest X-ray at 6 weeks"],
        "red_flags": ["Respiratory rate > 30", "SpO2 < 90%", "Systolic BP < 90"],
    },
    "I48.0": {  # Atrial fibrillation
        "anticoagulation": ["Apixaban", "Rivaroxaban", "Warfarin"],
        "rate_control": ["Metoprolol", "Diltiazem"],
        "monitoring": ["INR if on Warfarin", "Renal function q3-6mo"],
    },
    "E11": {  # Type 2 Diabetes
        "first_line": ["Metformin"],
        "add_on": ["GLP-1 agonist", "SGLT2 inhibitor"],
        "monitoring": ["HbA1c q3mo", "Annual foot exam", "Annual retinal exam"],
    },
}


# ── LangChain Clinical Decision Support Tools ─────────────────────────────────

@tool
def check_drug_guideline_adherence(diagnosis_code: str, proposed_drug: str) -> dict:
    """
    Check if proposed drug aligns with evidence-based clinical guidelines.

    Args:
        diagnosis_code: ICD-10 code (e.g., 'J18.9')
        proposed_drug: Drug name to check (may include dose/route, e.g. 'Azithromycin 500mg')

    Returns:
        Dictionary with guideline adherence status and recommendations

    FIX #1: The original code checked `drug_lower in drug.lower()` where `drug` iterated
    the guideline list — this tested whether the full proposed string (e.g. "azithromycin 500mg")
    was a substring of the short guideline entry ("azithromycin"), which always fails.
    Correct direction: check whether the guideline entry name is a substring of the
    proposed drug string (strip dose/route first for robustness).
    """
    # Resolve guideline: try full code, base code, then prefix match
    guideline = (
        CLINICAL_GUIDELINES.get(diagnosis_code)
        or CLINICAL_GUIDELINES.get(diagnosis_code.split(".")[0])
    )
    if not guideline:
        for key in CLINICAL_GUIDELINES:
            if diagnosis_code.startswith(key) or key.startswith(diagnosis_code.split(".")[0]):
                guideline = CLINICAL_GUIDELINES[key]
                break

    if not guideline:
        return {
            "guideline_available": False,
            "adherence": "UNKNOWN",
            "message": f"No guideline found for {diagnosis_code}",
        }

    # Normalise: lowercase the proposed drug string so matching is case-insensitive
    proposed_lower = proposed_drug.lower()

    # FIX #1 — correct containment direction:
    # Check if ANY guideline drug name appears *within* the proposed drug string.
    # e.g. guideline entry "azithromycin" IS contained in proposed "azithromycin 500mg" ✓
    # The old code tested: "azithromycin 500mg" in "azithromycin" — always False ✗

    def _matches(guideline_drug: str) -> bool:
        return guideline_drug.lower() in proposed_lower

    if any(_matches(g) for g in guideline.get("first_line", [])):
        return {
            "guideline_available": True,
            "adherence": "FIRST_LINE",
            "message": f"{proposed_drug} is first-line therapy per guidelines",
            "alternatives": guideline.get("second_line", []),
        }

    if any(_matches(g) for g in guideline.get("second_line", [])):
        return {
            "guideline_available": True,
            "adherence": "SECOND_LINE",
            "message": f"{proposed_drug} is acceptable second-line therapy",
            "preferred": guideline.get("first_line", []),
        }

    if any(_matches(g) for g in guideline.get("inpatient", [])):
        return {
            "guideline_available": True,
            "adherence": "INPATIENT_APPROPRIATE",
            "message": f"{proposed_drug} is appropriate for inpatient management",
        }

    # Also check anticoagulation / add_on / rate_control categories
    for category in ("anticoagulation", "add_on", "rate_control"):
        if any(_matches(g) for g in guideline.get(category, [])):
            return {
                "guideline_available": True,
                "adherence": "GUIDELINE_APPROPRIATE",
                "message": f"{proposed_drug} is appropriate per {category} guidelines",
            }

    return {
        "guideline_available": True,
        "adherence": "OFF_GUIDELINE",
        "message": f"{proposed_drug} not in standard guidelines for {diagnosis_code}",
        "recommended": guideline.get("first_line", []),
    }


@tool
def check_allergy_contraindications(
    proposed_drugs: List[str],
    allergies: List[dict],
    current_medications: List[dict],
) -> dict:
    """
    Check proposed drugs against patient allergies and known drug-drug interactions.

    FIX #5 (in main.py caller): cross-reactivity with severe allergy now correctly
    sets safety_flag = CONTRAINDICATED (see logic below — cross_reactivity entries
    with severe/anaphylaxis severity are now included in the CONTRAINDICATED check).
    """
    CROSS_REACTIVITY: Dict[str, List[str]] = {
        "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "ceftriaxone",
                       "cefazolin", "cephalexin", "cefepime"],
        "sulfa": ["trimethoprim-sulfamethoxazole", "sulfamethoxazole"],
        "aspirin": ["ibuprofen", "naproxen", "diclofenac", "ketorolac"],
    }

    DDI_TABLE: List[Tuple[str, str, str]] = [
        ("azithromycin", "warfarin",   "MAJOR: Azithromycin inhibits CYP3A4 — may significantly raise INR; monitor INR closely"),
        ("azithromycin", "apixaban",   "MODERATE: Azithromycin may increase apixaban levels"),
        ("levofloxacin", "warfarin",   "MAJOR: Fluoroquinolones may potentiate warfarin anticoagulation; monitor INR"),
        ("moxifloxacin", "warfarin",   "MAJOR: Fluoroquinolones may potentiate warfarin anticoagulation; monitor INR"),
        ("ceftriaxone",  "warfarin",   "MINOR: Limited interaction; routine INR monitoring recommended"),
        ("metformin",    "contrast",   "MODERATE: Hold metformin 48h before/after iodinated contrast"),
        ("prednisone",   "warfarin",   "MODERATE: Corticosteroids may increase or decrease INR"),
        ("doxycycline",  "warfarin",   "MODERATE: May potentiate warfarin; monitor INR"),
    ]

    allergy_alerts = []
    interaction_alerts = []

    patient_allergens = {a.get("substance", "").lower() for a in allergies}
    patient_allergen_severities = {
        a.get("substance", "").lower(): a.get("severity", "unknown")
        for a in allergies
    }

    for drug in proposed_drugs:
        drug_lower = drug.lower()

        # Direct allergy match
        for allergen in patient_allergens:
            if allergen in drug_lower or drug_lower in allergen:
                allergy_alerts.append({
                    "drug": drug,
                    "allergen": allergen,
                    "severity": patient_allergen_severities.get(allergen, "unknown"),
                    "alert": f"CONTRAINDICATED: Patient has documented {allergen} allergy "
                             f"({patient_allergen_severities.get(allergen, 'unknown')} severity)",
                })

        # Cross-reactivity check
        for allergen, cross_react_drugs in CROSS_REACTIVITY.items():
            if allergen in patient_allergens:
                if any(cr in drug_lower for cr in cross_react_drugs):
                    if allergen not in drug_lower:
                        allergy_alerts.append({
                            "drug": drug,
                            "allergen": allergen,
                            "cross_reactivity": True,
                            "severity": patient_allergen_severities.get(allergen, "unknown"),
                            "alert": f"CROSS-REACTIVITY RISK: {drug} may cross-react with documented "
                                     f"{allergen} allergy — verify tolerance before prescribing",
                        })

        # Drug-drug interaction check
        for existing_med in current_medications:
            existing_lower = existing_med.get("drug", "").lower()
            for proposed_frag, existing_frag, warning in DDI_TABLE:
                if proposed_frag in drug_lower and existing_frag in existing_lower:
                    interaction_alerts.append({
                        "proposed_drug": drug,
                        "existing_drug": existing_med.get("drug"),
                        "warning": warning,
                    })

    # FIX #5 — include cross-reactivity alerts in CONTRAINDICATED check when severity is severe/anaphylaxis.
    # Original code excluded cross_reactivity=True alerts from this check, allowing severe
    # penicillin allergy + cephalosporin cross-reactivity to only produce ALLERGY_RISK, not CONTRAINDICATED.
    severe_severities = ("severe", "anaphylaxis")

    direct_contraindicated = any(
        a.get("severity") in severe_severities
        for a in allergy_alerts
        if not a.get("cross_reactivity")
    )
    cross_react_contraindicated = any(
        a.get("severity") in severe_severities and a.get("cross_reactivity")
        for a in allergy_alerts
    )

    safety_flag = "SAFE"
    if direct_contraindicated or cross_react_contraindicated:
        safety_flag = "CONTRAINDICATED"
    elif allergy_alerts:
        safety_flag = "ALLERGY_RISK"
    elif interaction_alerts:
        safety_flag = "INTERACTION_WARNING"

    return {
        "allergy_alerts": allergy_alerts,
        "interaction_alerts": interaction_alerts,
        "safety_flag": safety_flag,
        "summary": (
            f"{len(allergy_alerts)} allergy alert(s), "
            f"{len(interaction_alerts)} drug-drug interaction(s) detected"
        ),
    }


@tool
def calculate_cha2ds2_vasc(
    age: int,
    gender: str,
    has_chf: bool,
    has_hypertension: bool,
    has_diabetes: bool,
    has_stroke_history: bool,
    has_vascular_disease: bool,
) -> dict:
    """
    Calculate CHA2DS2-VASc score for stroke risk in atrial fibrillation.
    """
    score = 0
    if has_chf:        score += 1
    if has_hypertension: score += 1
    if age >= 75:      score += 2
    elif age >= 65:    score += 1
    if has_diabetes:   score += 1
    if has_stroke_history: score += 2
    if has_vascular_disease: score += 1
    if gender.lower() in ("female", "f"): score += 1

    if score == 0:
        recommendation = "No anticoagulation (male) or consider anticoagulation (female)"
        risk_category = "LOW"
    elif score == 1:
        recommendation = "Consider anticoagulation - shared decision making"
        risk_category = "MODERATE"
    else:
        recommendation = "Anticoagulation recommended unless contraindicated"
        risk_category = "HIGH"

    annual_stroke_risks = [0, 1.3, 2.2, 3.2, 4.0, 6.7, 9.8, 9.6, 6.7, 15.2]

    return {
        "score": score,
        "risk_category": risk_category,
        "recommendation": recommendation,
        "annual_stroke_risk_percent": annual_stroke_risks[min(score, 9)],
    }


@tool
def estimate_qaly_impact(
    baseline_quality_of_life: float,
    treatment_efficacy: float,
    treatment_duration_years: float,
    side_effect_qol_reduction: float = 0.0,
) -> dict:
    """
    Estimate Quality-Adjusted Life Years (QALY) impact of treatment.
    """
    qaly_without = baseline_quality_of_life * treatment_duration_years
    treated_qol = min(
        1.0,
        baseline_quality_of_life + treatment_efficacy - side_effect_qol_reduction,
    )
    qaly_with = treated_qol * treatment_duration_years
    qaly_gained = qaly_with - qaly_without
    value_at_threshold = qaly_gained * 100_000

    return {
        "qaly_gained": round(qaly_gained, 3),
        "qaly_without_treatment": round(qaly_without, 3),
        "qaly_with_treatment": round(qaly_with, 3),
        "value_at_100k_threshold_usd": round(value_at_threshold, 2),
        "cost_effective_if_treatment_cost_below_usd": round(value_at_threshold, 2),
    }
 
 
EXTREME_VALUE_THRESHOLDS = {
    "wbc":               12.0,   # >12 = elevated; >15 = concerning
    "crp":               50.0,   # >50 = significant inflammation
    "creatinine":        1.5,    # >1.5 = renal impairment
    "glucose":           180.0,  # >180 = hyperglycaemia
    "critical_lab_count": 1,     # any critical flag = high perturbation
}
 
 
def _gradient_search(
    model,
    feature_vector: List[float],
    feat_idx: int,
    baseline_pred: float,
    direction: str,
    max_pct: float = 0.60,
    step_pct: float = 0.05,
) -> tuple[float | None, float | None]:
    """
    Walk perturbation from step_pct up to max_pct in steps.
    Returns (first_pct_that_caused_change, new_pred_at_that_pct)
    or (None, None) if truly insensitive across the full range.
    """
    baseline_value = feature_vector[feat_idx]
    for pct in np.arange(step_pct, max_pct + step_pct, step_pct):
        fv = feature_vector.copy()
        if direction == "decrease":
            fv[feat_idx] = baseline_value * (1.0 - pct)
        else:
            fv[feat_idx] = baseline_value * (1.0 + pct)
        pred = float(model.predict_proba(np.array([fv]))[0][1])
        if abs(pred - baseline_pred) > 0.001:
            return round(float(pct * 100), 1), round(pred, 4)
    return None, None

# ── Analysis Functions ────────────────────────────────────────────────────────

def perform_sensitivity_analysis(
    feature_vector: List[float],
    feature_dict: Dict,
    models: Dict,
    feature_names: List[str],
) -> List[Dict]:
    """
    One-way sensitivity analysis with adaptive perturbation.
 
    Bug 2 Fix:
    - Features with extreme values (above clinical thresholds) use 40%
      perturbation instead of 20% to cross XGBoost tree boundaries
    - If still insensitive, a gradient search walks from 5% to 60% to find
      the first threshold that actually moves the prediction
    - insensitive_note now reports the minimum perturbation needed (if any)
    """
    if "mortality_30d" not in models:
        return []
 
    model = models["mortality_30d"]
    X = np.array([feature_vector])
    baseline_pred = float(model.predict_proba(X)[0][1])
 
    modifiable_features = {
        "wbc":                {"modifiable": True,  "intervention": "Antibiotic therapy"},
        "crp":                {"modifiable": True,  "intervention": "Anti-inflammatory treatment"},
        "creatinine":         {"modifiable": True,  "intervention": "Hydration, medication adjustment"},
        "albumin":            {"modifiable": True,  "intervention": "Nutritional support"},
        "glucose":            {"modifiable": True,  "intervention": "Glycemic control"},
        "critical_lab_count": {"modifiable": True,  "intervention": "Targeted interventions"},
        "age":                {"modifiable": False, "intervention": None},
        "comorbidity_count":  {"modifiable": False, "intervention": "Chronic disease management"},
    }
 
    improvement_direction = {
        "wbc": "decrease", "crp": "decrease", "creatinine": "decrease",
        "glucose": "decrease", "critical_lab_count": "decrease",
        "albumin": "increase", "hemoglobin": "increase",
        "age": "decrease", "comorbidity_count": "decrease",
    }
 
    sensitivity_results = []
 
    for feat_name, feat_meta in modifiable_features.items():
        if feat_name not in feature_dict or feat_name not in feature_names:
            continue
 
        feat_idx      = feature_names.index(feat_name)
        baseline_val  = feature_vector[feat_idx]
        direction     = improvement_direction.get(feat_name, "decrease")
 
        # --- Bug 2 fix: adaptive perturbation ---
        threshold = EXTREME_VALUE_THRESHOLDS.get(feat_name)
        if threshold is not None and baseline_val > threshold:
            primary_pct = 0.40   # extreme value → 40%
        else:
            primary_pct = 0.20   # normal range → 20%
 
        def _perturb(pct: float, improve: bool) -> float:
            fv = feature_vector.copy()
            if improve:
                fv[feat_idx] = baseline_val * (1.0 - pct) if direction == "decrease" else baseline_val * (1.0 + pct)
            else:
                fv[feat_idx] = baseline_val * (1.0 + pct) if direction == "decrease" else baseline_val * (1.0 - pct)
            return float(model.predict_proba(np.array([fv]))[0][1])
 
        improved_pred  = _perturb(primary_pct, improve=True)
        worsened_pred  = _perturb(primary_pct, improve=False)
        sensitivity_mag = abs(round((worsened_pred - improved_pred) * 100, 2))


        # Bug 2 bonus: sanity check — "lower is better" features should
        # reduce mortality when improved. Log if model produces the reverse.
        lower_is_better_features = {"wbc", "crp", "creatinine", "glucose", "critical_lab_count"}
        if feat_name in lower_is_better_features and improved_pred > baseline_pred:
            print(
                f"  [sensitivity warning] {feat_name}: improving this feature "
                f"increased predicted mortality ({baseline_pred:.3f} → {improved_pred:.3f}). "
                f"Likely a synthetic-data artifact in the mortality_30d model — "
                f"do not surface this as a clinical finding."
            )
 
        # --- Bug 2 fix: gradient search fallback ---
        model_sensitivity = "SENSITIVE" if sensitivity_mag > 0 else "INSENSITIVE"
        insensitive_note  = None
        min_pct_needed    = None
 
        if model_sensitivity == "INSENSITIVE":
            threshold_pct, _ = _gradient_search(
                model, feature_vector, feat_idx, baseline_pred,
                direction, max_pct=0.60, step_pct=0.05,
            )
            if threshold_pct is not None:
                min_pct_needed   = threshold_pct
                model_sensitivity = "INSENSITIVE_AT_PRIMARY"
                insensitive_note  = (
                    f"XGBoost tree boundary not crossed at {primary_pct*100:.0f}% perturbation. "
                    f"First detectable response at ~{threshold_pct:.0f}% change — "
                    f"clinical relevance exists; effect appears at larger perturbations."
                )
            else:
                insensitive_note = (
                    f"XGBoost model shows no sensitivity to {feat_name} across 5-60% perturbation range. "
                    f"Feature may not be a split variable in the mortality_30d tree ensemble. "
                    f"Clinical relevance still applies — consider this a model limitation, not a clinical finding."
                )
 
        sensitivity_results.append({
            "feature_name":                     feat_name,
            "baseline_value":                   round(baseline_val, 2),
            "perturbation_pct_used":            int(primary_pct * 100),
            "improvement_direction":             direction,
            "risk_impact_if_improved":          {
                "mortality_30d_change": round((improved_pred - baseline_pred) * 100, 2),
                "new_risk":             round(improved_pred, 4),
            },
            "risk_impact_if_worsened":          {
                "mortality_30d_change": round((worsened_pred - baseline_pred) * 100, 2),
                "new_risk":             round(worsened_pred, 4),
            },
            # Keep old key names for backwards compat with existing callers
            "risk_impact_if_improved_20_percent": {
                "mortality_30d_change": round((improved_pred - baseline_pred) * 100, 2),
                "new_risk":             round(improved_pred, 4),
            },
            "risk_impact_if_worsened_20_percent": {
                "mortality_30d_change": round((worsened_pred - baseline_pred) * 100, 2),
                "new_risk":             round(worsened_pred, 4),
            },
            "modifiable":              feat_meta["modifiable"],
            "clinical_intervention":   feat_meta["intervention"],
            "sensitivity_magnitude":   sensitivity_mag,
            "model_sensitivity":       model_sensitivity,
            "min_perturbation_pct_for_response": min_pct_needed,
            "insensitive_note":        insensitive_note,
        })
 
    # Sort: modifiable first (by magnitude), then non-modifiable
    modifiable_r     = sorted([r for r in sensitivity_results if r["modifiable"]],
                               key=lambda x: -x["sensitivity_magnitude"])
    non_modifiable_r = sorted([r for r in sensitivity_results if not r["modifiable"]],
                               key=lambda x: -x["sensitivity_magnitude"])
    return (modifiable_r + non_modifiable_r)[:6]


def analyze_cost_effectiveness(scenarios: List[Dict], patient_age: int) -> Dict:
    """
    Incremental cost-effectiveness analysis (ICEA).

    FIX #4 — the "no treatment" baseline (option_id == "C" or cost == 0 with no drugs)
              is excluded from most_cost_effective. It is retained in the scenarios list
              as a comparator but is clearly labelled as "BASELINE_COMPARATOR" and will
              never be returned as the recommended cost-effective option.
    FIX #10 — imputed costs are propagated back so callers can use them.
    """
    if not scenarios:
        return {}

    DISCOUNT_RATE = 0.03
    life_expectancy_remaining = max(1, 85 - patient_age)

    discounted_ly = sum(
        1 / (1 + DISCOUNT_RATE) ** t
        for t in range(1, life_expectancy_remaining + 1)
    )

    cea_results = []

    for scenario in scenarios:
        preds = scenario.get("predictions", {})

        mortality_risk_30d = preds.get("mortality_risk_30d", 0.05)
        complication_risk  = preds.get("complication_risk", 0.10)
        recovery_prob_7d   = preds.get("recovery_probability_7d", 0.70)

        prob_surviving_30d = 1.0 - mortality_risk_30d
        survival_adjusted_ly = discounted_ly * prob_surviving_30d

        baseline_qol   = 0.65
        recovery_uplift = recovery_prob_7d * 0.20
        complication_penalty = complication_risk * 0.10
        treated_qol = min(0.95, max(0.10, baseline_qol + recovery_uplift - complication_penalty))

        qalys = round(treated_qol * survival_adjusted_ly, 2)

        cost        = scenario.get("estimated_cost_usd")
        cost_source = scenario.get("cost_source")   # already set upstream if via /simulate
        is_baseline = (
            scenario.get("option_id") == "C"
            or (not scenario.get("drugs") and not scenario.get("interventions"))
        )
 
        if cost is not None and cost_source:
            # Resolved upstream — trust it
            cost = float(cost)
        else:
            # Fallback for stream path (stream_endpoints builds scenarios inline)
            if is_baseline:
                cost, cost_source = 0.0, "zero"
            elif cost is None or cost == 0:
                if any("hospitalization" in i.lower() for i in scenario.get("interventions", [])):
                    cost, cost_source = 15_000.0, "imputed"
                else:
                    cost, cost_source = 500.0, "imputed"
            else:
                cost_source = "provided"

        result = {
            "option_id": scenario["option_id"],
            "label": scenario["label"],
            "estimated_cost_usd": cost,
            "cost_source": cost_source,
            "is_baseline_comparator": is_baseline,
            "survival_probability_30d": round(prob_surviving_30d, 3),
            "quality_of_life_score": round(treated_qol, 3),
            "estimated_qalys": qalys,
            "cost_per_qaly": round(cost / max(qalys, 0.01), 2) if qalys > 0 else 999_999,
        }
        cea_results.append(result)

    # FIX #4 — only consider actual treatment options (not baseline) for cost-effectiveness ranking
    treatment_results = [r for r in cea_results if not r.get("is_baseline_comparator")]
    treatment_results.sort(key=lambda x: x["cost_per_qaly"])

    # Build a set of contraindicated option IDs to pass into CEA
    contraindicated_ids = {
        s["option_id"] for s in scenarios
        if (s.get("safety_check") or {}).get("safety_flag") == "CONTRAINDICATED"
    }

    for result in cea_results:
        cpq = result["cost_per_qaly"]
        if result.get("is_baseline_comparator"):
            result["cost_effective"] = "BASELINE_COMPARATOR"
        elif result["option_id"] in contraindicated_ids:
            result["cost_effective"] = "CONTRAINDICATED"
        elif cpq < 100_000:
            result["cost_effective"] = True
        elif cpq < 150_000:
            result["cost_effective"] = "BORDERLINE"
        else:
            result["cost_effective"] = False

    # Sort for display: treatments first (by cost/QALY), then baseline
    sorted_display = treatment_results + [r for r in cea_results if r.get("is_baseline_comparator")]

    return {
        "scenarios": sorted_display,
        # FIX #4 — most_cost_effective is the best *treatment* option, never Option C
        "most_cost_effective": treatment_results[0]["option_id"] if treatment_results else None,
        "threshold_used_usd_per_qaly": 100_000,
        "discount_rate_used": DISCOUNT_RATE,
        "interpretation": (
            "Incremental cost-effectiveness analysis. QALYs discounted at 3% annually. "
            "Survival-adjusted for 30-day mortality risk. "
            "Standard threshold: $100,000/QALY in US healthcare. "
            "Option C (no treatment) is the baseline comparator — not a treatment recommendation. "
            "Costs marked 'imputed' are estimated — provide actual costs for precise analysis."
        ),
    }


def _extract_ddi_monitoring_context(scenarios: list, recommended_id: str) -> str:
    """
    Build specific INR/DDI monitoring instructions from the recommended
    scenario's safety_check interaction_alerts.
    Bug 4 fix: generates concrete clinical actions instead of generic language.
    """
    rec = next((s for s in scenarios if s.get("option_id") == recommended_id), None)
    if not rec:
        return ""
 
    safety  = rec.get("safety_check") or {}
    ia_list = safety.get("interaction_alerts", [])
    aa_list = safety.get("allergy_alerts", [])
    lines   = []
 
    for ia in ia_list:
        proposed = ia.get("proposed_drug", "")
        existing = ia.get("existing_drug", "")
        warning  = ia.get("warning", "")
 
        if "warfarin" in existing.lower():
            if "azithromycin" in proposed.lower():
                lines.append(
                    f"DDI ({proposed} \u2194 {existing}): {warning}. "
                    "REQUIRED ACTIONS: (1) Check baseline INR before prescribing. "
                    "(2) Recheck INR at 48-72 hours after starting azithromycin. "
                    "(3) Hold or reduce warfarin dose if INR >3.5. "
                    "(4) Target INR 2.0-3.0 for atrial fibrillation."
                )
            elif any(q in proposed.lower() for q in ("levofloxacin", "moxifloxacin")):
                lines.append(
                    f"DDI ({proposed} \u2194 {existing}): {warning}. "
                    "REQUIRED ACTIONS: (1) Check baseline INR before prescribing. "
                    "(2) Recheck INR at 3-5 days. "
                    "(3) Consider empirical 20-25% warfarin dose reduction in high-risk patients."
                )
            elif "ceftriaxone" in proposed.lower():
                lines.append(
                    f"DDI ({proposed} \u2194 {existing}): {warning}. "
                    "REQUIRED ACTION: Recheck INR within 5-7 days of starting ceftriaxone."
                )
            else:
                lines.append(
                    f"DDI ({proposed} \u2194 {existing}): {warning}. "
                    "REQUIRED ACTION: Monitor INR closely during co-administration."
                )
        elif "metformin" in (existing + proposed).lower():
            lines.append(
                f"DDI ({proposed} \u2194 {existing}): {warning}. "
                "REQUIRED ACTION: Hold metformin 48h before and after iodinated contrast."
            )
        else:
            lines.append(f"DDI ({proposed} \u2194 {existing}): {warning}.")
 
    for aa in aa_list:
        if aa.get("cross_reactivity"):
            lines.append(
                f"Cross-reactivity alert ({aa.get('drug')} \u2194 "
                f"documented {aa.get('allergen')} allergy, {aa.get('severity')} severity): "
                "verify tolerance before prescribing — consider allergy skin testing or "
                "graded challenge if clinically necessary."
            )
 
    return "\n".join(lines) if lines else ""
 
 
def build_enhanced_llm_narrative(
    patient_state: dict,
    diagnosis: str,
    diagnosis_code: Optional[str],
    scenarios: List[dict],
    recommended_option: str,
    risk_profile: str,
    llm: Optional[ChatGoogleGenerativeAI],
    llm_ready: bool,
    sensitivity_top_3: Optional[List[Dict]] = None,
    cost_effectiveness: Optional[Dict] = None,
) -> str:
    """
    Generate evidence-based clinical narrative.
    Bug 4 fix: MONITORING section now receives DDI-specific context and is
    instructed to produce concrete clinical actions (INR targets, timing,
    dose thresholds) rather than generic language.
    """
    demo    = patient_state.get("demographics", {})
    age     = demo.get("age", "?")
    gender  = demo.get("gender", "patient")
    best    = next((s for s in scenarios if s["option_id"] == recommended_option),
                   scenarios[0] if scenarios else {})
 
    modifiable_sensitivity = [
        s for s in (sensitivity_top_3 or [])
        if s.get("modifiable", False)
    ]
 
    # Bug 4 fix: build DDI monitoring context
    ddi_monitoring = _extract_ddi_monitoring_context(scenarios, recommended_option)
    monitoring_block = (
        f"\n\nDrug interaction monitoring requirements for the recommended option:\n{ddi_monitoring}"
        if ddi_monitoring
        else "\n\nNo significant drug interactions detected for the recommended option."
    )
 
    if not llm_ready or not llm:
        rec_label = best.get("label", recommended_option)
        rec_7d    = best.get("predictions", {}).get("recovery_probability_7d", 0)
        rec_mort  = best.get("predictions", {}).get("mortality_risk_30d", 0)
 
        narrative = (
            f"For this {age}-year-old {gender} with {diagnosis} ({risk_profile} risk), "
            f"{rec_label} offers {rec_7d:.0%} 7-day recovery probability and "
            f"{rec_mort:.0%} 30-day mortality risk. "
        )
        if modifiable_sensitivity:
            top    = modifiable_sensitivity[0]
            change = top.get("risk_impact_if_improved_20_percent", {})
            narrative += (
                f"Most modifiable risk factor: {top['feature_name']} "
                f"(improvement reduces mortality by "
                f"{abs(change.get('mortality_30d_change', 0)):.1f}% points). "
            )
        if cost_effectiveness:
            most_ce = cost_effectiveness.get("most_cost_effective")
            if most_ce == recommended_option:
                narrative += "This option is also most cost-effective. "
        if ddi_monitoring:
            # Include first DDI action in fallback narrative
            first_line = ddi_monitoring.split("\n")[0]
            narrative += f"Key monitoring: {first_line} "
        narrative += "Evidence-based decision support — clinical judgment required."
        return narrative
 
    # Build scenario summary lines for the LLM
    scenario_lines = []
    for s in scenarios:
        if s["option_id"] == "C":
            continue
        preds           = s.get("predictions", {})
        guideline_raw   = s.get("guideline_adherence", {})
        guideline       = (guideline_raw[0] if isinstance(guideline_raw, list) and guideline_raw
                           else guideline_raw if isinstance(guideline_raw, dict) else {})
 
        adherence_note = ""
        if guideline:
            adherence = guideline.get("adherence", "")
            adherence_note = {
                "FIRST_LINE":           " [FIRST-LINE per guidelines]",
                "INPATIENT_APPROPRIATE": " [INPATIENT guideline]",
                "SECOND_LINE":           " [SECOND-LINE]",
                "OFF_GUIDELINE":         " [OFF-GUIDELINE]",
            }.get(adherence, "")
 
        safety_flag = (s.get("safety_check") or {}).get("safety_flag", "SAFE")
        safety_note = {
            "CONTRAINDICATED":     " [CONTRAINDICATED — excluded from recommendation]",
            "ALLERGY_RISK":        " [ALLERGY RISK — use with caution]",
            "INTERACTION_WARNING": " [DDI warning — see monitoring]",
        }.get(safety_flag, "")
 
        scenario_lines.append(
            f"Option {s['option_id']} ({s['label']}){adherence_note}{safety_note}: "
            f"7d recovery {preds.get('recovery_probability_7d', 0):.0%}, "
            f"30d mortality {preds.get('mortality_risk_30d', 0):.0%}, "
            f"30d readmission {preds.get('readmission_risk_30d', 0):.0%}"
        )
 
    conditions          = patient_state.get("active_conditions", [])
    comorbidity_summary = ", ".join(c.get("display", "") for c in conditions[:3])
 
    sensitivity_context = ""
    if modifiable_sensitivity:
        sensitivity_context = "\n\nKey MODIFIABLE risk factors (clinician can intervene):\n" + "\n".join([
            f"- {s['feature_name']} ({s.get('clinical_intervention', 'intervention available')}): "
            f"20% improvement → "
            f"{abs(s.get('risk_impact_if_improved_20_percent', {}).get('mortality_30d_change', 0)):.1f}% mortality reduction"
            for s in modifiable_sensitivity[:3]
        ])
 
    cost_context = ""
    if cost_effectiveness:
        most_ce      = cost_effectiveness.get("most_cost_effective")
        cost_context = (
            f"\n\nCost-effectiveness: Option {most_ce} is most cost-effective "
            "among treatment options at current willingness-to-pay threshold."
        )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical decision support system generating structured clinical decision support notes. "
         "Use the following format exactly:\n\n"
         "IMPRESSION: One sentence summarizing patient risk profile and primary diagnosis.\n\n"
         "SAFETY: One sentence. If any option is CONTRAINDICATED, explicitly name it, state why "
         "(allergy/cross-reactivity/DDI), and confirm it is excluded from recommendation.\n\n"
         "RECOMMENDATION: One to two sentences. State the recommended option, cite its key outcome "
         "metrics (7d recovery, 30d mortality, readmission), and note guideline adherence status.\n\n"
         "MONITORING: Two to three sentences. You MUST use the 'Drug interaction monitoring requirements' "
         "provided below. Reference the specific drug names, the specific INR targets and recheck timing, "
         "and the specific dose-adjustment thresholds. "
         "FORBIDDEN phrases: 'monitor closely', 'monitor for drug interactions', 'close monitoring'. "
         "REQUIRED format example: 'Check baseline INR before starting azithromycin; recheck at 48-72h; "
         "hold warfarin if INR >3.5; target INR 2.0-3.0 for atrial fibrillation.'\n\n"
         "MODIFIABLE RISK FACTORS: One sentence. Name 1-2 specific modifiable lab factors with their "
         "quantified mortality impact from the sensitivity data. Do NOT mention age.\n\n"
         "CRITICAL RULES:\n"
         "- CONTRAINDICATED options must never appear as recommendations or alternatives.\n"
         "- All percentages must come from the simulation data provided — do not invent numbers.\n"
         "- MONITORING must contain the exact drug names and exact thresholds from the DDI context.\n"
         "- End with exactly: 'This is AI-generated decision support requiring physician validation.'"),
        ("human",
         f"Patient: {age}y {gender}. "
         f"Diagnosis: {diagnosis} (ICD-10: {diagnosis_code or 'not specified'}). "
         f"Comorbidities: {comorbidity_summary or 'none documented'}. "
         f"Risk profile: {risk_profile}.\n\n"
         f"Treatment scenarios:\n" + "\n".join(scenario_lines) + "\n\n"
         f"Recommended: Option {recommended_option}"
         f"{sensitivity_context}"
         f"{cost_context}"
         f"{monitoring_block}\n\n"
         "Generate the structured clinical decision support note."),
    ])
 
    chain = prompt | llm | StrOutputParser()
    try:
        narrative = chain.invoke({})
        if "AI-generated" not in narrative:
            narrative += " This is AI-generated decision support requiring physician validation."
        return narrative
    except Exception as e:
        print(f"  LLM narrative failed: {e}")
        return (
            f"Digital Twin simulation recommends Option {recommended_option} "
            f"based on {risk_profile} risk profile. "
            "Evidence-based decision support — physician validation required."
        )


# ── FHIR CarePlan Builder ─────────────────────────────────────────────────────

def build_enhanced_fhir_care_plan(
    patient_id: str,
    recommended_option: TreatmentOption,
    narrative: str,
    predicted_recovery: float,
    prediction_confidence: str,
    diagnosis_code: Optional[str],
    feature_attribution: List[Dict],
    model_version: str = "2.0.0",
) -> dict:
    """
    Build comprehensive FHIR R4 CarePlan with provenance and confidence tracking.
    """
    plan_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    activities = []

    for drug in recommended_option.drugs:
        activities.append({
            "detail": {
                "kind": "MedicationRequest",
                "code": {"text": drug},
                "status": "scheduled",
                "description": f"Proposed: {drug}",
                "instantiatesCanonical": (
                    f"#guideline-{diagnosis_code}" if diagnosis_code else None
                ),
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

    if diagnosis_code and diagnosis_code in CLINICAL_GUIDELINES:
        for monitoring in CLINICAL_GUIDELINES[diagnosis_code].get("monitoring", []):
            activities.append({
                "detail": {
                    "kind": "ServiceRequest",
                    "code": {"text": monitoring},
                    "status": "scheduled",
                    "description": f"Follow-up monitoring: {monitoring}",
                }
            })

    risk_attribution_extension = [
        {
            "url": "http://meditwin.ai/fhir/StructureDefinition/risk-factor",
            "extension": [
                {"url": "factor",       "valueString":  attr["feature"]},
                {"url": "contribution", "valueString":  attr["contribution"]},
                {"url": "direction",    "valueString":  attr["direction"]},
                {"url": "importance",   "valueDecimal": round(attr.get("importance_score", 0), 4)},
            ],
        }
        for attr in feature_attribution[:5]
    ]

    return {
        "resourceType": "CarePlan",
        "id": plan_id,
        "meta": {
            "versionId": "1",
            "lastUpdated": timestamp,
            "profile": ["http://hl7.org/fhir/StructureDefinition/CarePlan"],
            "tag": [
                {
                    "system": "http://meditwin.ai/fhir/CodeSystem/ai-generated",
                    "code": "digital-twin-simulation",
                    "display": "AI Digital Twin Simulation",
                },
                {
                    "system": "http://meditwin.ai/fhir/CodeSystem/confidence",
                    "code": prediction_confidence.lower(),
                    "display": f"Model Confidence: {prediction_confidence}",
                },
            ],
        },
        "extension": [
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/ai-provenance",
                "extension": [
                    {"url": "model",      "valueString":   f"MediTwin Digital Twin v{model_version}"},
                    {"url": "algorithm",  "valueString":   "XGBoost"},
                    {"url": "timestamp",  "valueDateTime": timestamp},
                    {"url": "confidence", "valueString":   prediction_confidence},
                ],
            },
            {
                "url": "http://meditwin.ai/fhir/StructureDefinition/predicted-outcome",
                "extension": [
                    {"url": "recovery-probability-7d", "valueDecimal": round(predicted_recovery, 4)},
                    {"url": "confidence-interval",     "valueString":  "95% CI available in source"},
                ],
            },
        ] + risk_attribution_extension,
        "status": "active",
        "intent": "proposal",
        "subject": {"reference": f"Patient/{patient_id}"},
        "created": timestamp,
        "title": (
            f"AI-Recommended Treatment Plan — "
            f"Option {recommended_option.option_id}: {recommended_option.label}"
        ),
        "description": narrative,
        "activity": activities,
        "note": [
            {
                "authorString": "MediTwin Digital Twin Agent",
                "time": timestamp,
                "text": (
                    f"AI-generated treatment simulation with {prediction_confidence} confidence. "
                    f"Predicted 7-day recovery probability: {predicted_recovery:.0%}. "
                    "Based on XGBoost risk models trained on synthetic clinical data. "
                    "This is DECISION SUPPORT only - not a substitute for clinical judgment. "
                    "Physician review, patient consent, and clinical validation required before implementation."
                ),
            }
        ],
    }