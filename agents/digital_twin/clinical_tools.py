"""
clinical_tools.py — Clinical guidelines, LangChain decision support tools,
                    and analysis functions (sensitivity, cost-effectiveness,
                    narrative generation, FHIR CarePlan builder).
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
        "inpatient": ["Ceftriaxone IV + Azithromycin"],
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
        proposed_drug: Drug name to check

    Returns:
        Dictionary with guideline adherence status and recommendations
    """
    # Try full code first (e.g. "J18.9"), then base code (e.g. "J18"), then prefix
    guideline = (
        CLINICAL_GUIDELINES.get(diagnosis_code)
        or CLINICAL_GUIDELINES.get(diagnosis_code.split(".")[0])
    )
    if not guideline:
        # Try prefix match (e.g. "E11" matches "E11.9")
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

    drug_lower = proposed_drug.lower()

    if any(drug_lower in drug.lower() for drug in guideline.get("first_line", [])):
        return {
            "guideline_available": True,
            "adherence": "FIRST_LINE",
            "message": f"{proposed_drug} is first-line therapy per guidelines",
            "alternatives": guideline.get("second_line", []),
        }

    if any(drug_lower in drug.lower() for drug in guideline.get("second_line", [])):
        return {
            "guideline_available": True,
            "adherence": "SECOND_LINE",
            "message": f"{proposed_drug} is acceptable second-line therapy",
            "preferred": guideline.get("first_line", []),
        }

    if any(drug_lower in drug.lower() for drug in guideline.get("inpatient", [])):
        return {
            "guideline_available": True,
            "adherence": "INPATIENT_APPROPRIATE",
            "message": f"{proposed_drug} is appropriate for inpatient management",
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
    Check proposed drugs against patient allergies and known drug-drug interactions
    with existing medications (e.g. Azithromycin + Warfarin → raised INR).

    Args:
        proposed_drugs: List of drug names being considered
        allergies: Patient allergy list from patient_state
        current_medications: Active medication list from patient_state

    Returns:
        Dict with allergy_alerts, interaction_alerts, and overall safety_flag
    """
    # Cross-reactivity map: if patient is allergic to key, flag these drugs
    CROSS_REACTIVITY: Dict[str, List[str]] = {
        "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "ceftriaxone",
                       "cefazolin", "cephalexin", "cefepime"],  # ~1-2% cross-reactivity
        "sulfa": ["trimethoprim-sulfamethoxazole", "sulfamethoxazole"],
        "aspirin": ["ibuprofen", "naproxen", "diclofenac", "ketorolac"],
    }

    # Known drug-drug interactions with existing meds
    # Format: (proposed_drug_fragment, existing_drug_fragment) → warning
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

    # Build sets of patient allergens (lowercase)
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
                    # Avoid duplicate with direct allergy
                    if allergen not in drug_lower:
                        allergy_alerts.append({
                            "drug": drug,
                            "allergen": allergen,
                            "cross_reactivity": True,
                            "severity": patient_allergen_severities.get(allergen, "unknown"),
                            "alert": f"CROSS-REACTIVITY RISK: {drug} may cross-react with documented "
                                     f"{allergen} allergy — verify tolerance before prescribing",
                        })

        # Drug-drug interaction check against existing medications
        for existing_med in current_medications:
            existing_lower = existing_med.get("drug", "").lower()
            for proposed_frag, existing_frag, warning in DDI_TABLE:
                if proposed_frag in drug_lower and existing_frag in existing_lower:
                    interaction_alerts.append({
                        "proposed_drug": drug,
                        "existing_drug": existing_med.get("drug"),
                        "warning": warning,
                    })

    safety_flag = "SAFE"
    if any(a.get("severity") in ("severe", "anaphylaxis") for a in allergy_alerts
           if not a.get("cross_reactivity")):
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
    Used to guide anticoagulation decisions.

    Returns score and anticoagulation recommendation.
    """
    score = 0

    if has_chf:
        score += 1
    if has_hypertension:
        score += 1
    if age >= 75:
        score += 2
    elif age >= 65:
        score += 1
    if has_diabetes:
        score += 1
    if has_stroke_history:
        score += 2
    if has_vascular_disease:
        score += 1
    if gender.lower() in ("female", "f"):
        score += 1

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
    Used in cost-effectiveness analysis.

    Args:
        baseline_quality_of_life: 0.0 to 1.0 (1.0 = perfect health)
        treatment_efficacy: Expected improvement in QoL (0.0 to 1.0)
        treatment_duration_years: How long treatment lasts
        side_effect_qol_reduction: QoL reduction from side effects

    Returns:
        QALY gained and value at standard threshold
    """
    qaly_without = baseline_quality_of_life * treatment_duration_years
    treated_qol = min(
        1.0,
        baseline_quality_of_life + treatment_efficacy - side_effect_qol_reduction,
    )
    qaly_with = treated_qol * treatment_duration_years
    qaly_gained = qaly_with - qaly_without
    value_at_threshold = qaly_gained * 100_000  # $100k/QALY midpoint

    return {
        "qaly_gained": round(qaly_gained, 3),
        "qaly_without_treatment": round(qaly_without, 3),
        "qaly_with_treatment": round(qaly_with, 3),
        "value_at_100k_threshold_usd": round(value_at_threshold, 2),
        "cost_effective_if_treatment_cost_below_usd": round(value_at_threshold, 2),
    }


# ── Analysis Functions ────────────────────────────────────────────────────────

def perform_sensitivity_analysis(
    feature_vector: List[float],
    feature_dict: Dict,
    models: Dict,
    feature_names: List[str],
) -> List[Dict]:
    """
    One-way sensitivity analysis: how much does each key feature impact risk?
    Identifies which modifiable factors matter most.

    Args:
        feature_vector: Engineered feature list
        feature_dict: Raw feature dictionary
        models: Loaded XGBoost model registry
        feature_names: Ordered list of feature names (FEATURE_NAMES)
    """
    if "mortality_30d" not in models:
        return []

    model = models["mortality_30d"]
    baseline_pred = model.predict_proba(np.array([feature_vector]))[0][1]

    modifiable_features = {
        "wbc":                 {"modifiable": True,  "intervention": "Antibiotic therapy"},
        "crp":                 {"modifiable": True,  "intervention": "Anti-inflammatory treatment"},
        "creatinine":          {"modifiable": True,  "intervention": "Hydration, medication adjustment"},
        "albumin":             {"modifiable": True,  "intervention": "Nutritional support"},
        "glucose":             {"modifiable": True,  "intervention": "Glycemic control"},
        "critical_lab_count":  {"modifiable": True,  "intervention": "Targeted interventions"},
        "age":                 {"modifiable": False, "intervention": None},
        "comorbidity_count":   {"modifiable": False, "intervention": "Chronic disease management"},
    }

    # Features where lower is better (decrease by 10% = "improve")
    lower_is_better = {"wbc", "crp", "creatinine", "glucose", "critical_lab_count"}
    # Features where higher is better (increase by 10% = "improve")
    higher_is_better = {"albumin", "hemoglobin"}
    # Non-modifiable features: "improve" = younger (lower age), "worsen" = older (higher age)
    # We define improvement direction explicitly to avoid sign confusion
    improvement_direction = {
        "wbc":                "decrease",
        "crp":                "decrease",
        "creatinine":         "decrease",
        "glucose":            "decrease",
        "critical_lab_count": "decrease",
        "albumin":            "increase",
        "hemoglobin":         "increase",
        "age":                "decrease",  # "improvement" = younger patient (non-modifiable, for reference)
        "comorbidity_count":  "decrease",
    }

    sensitivity_results = []

    for feat_name, feat_meta in modifiable_features.items():
        if feat_name not in feature_dict or feat_name not in feature_names:
            continue

        feat_idx = feature_names.index(feat_name)
        baseline_value = feature_vector[feat_idx]

        direction = improvement_direction.get(feat_name, "decrease")

        # 10% improvement in the clinically correct direction
        improved_vector = feature_vector.copy()
        if direction == "decrease":
            improved_vector[feat_idx] = baseline_value * 0.9  # lower = better
        else:
            improved_vector[feat_idx] = baseline_value * 1.1  # higher = better
        improved_pred = model.predict_proba(np.array([improved_vector]))[0][1]

        # 10% worsening (opposite direction)
        worsened_vector = feature_vector.copy()
        if direction == "decrease":
            worsened_vector[feat_idx] = baseline_value * 1.1  # higher = worse
        else:
            worsened_vector[feat_idx] = baseline_value * 0.9  # lower = worse
        worsened_pred = model.predict_proba(np.array([worsened_vector]))[0][1]

        sensitivity_results.append({
            "feature_name": feat_name,
            "baseline_value": round(baseline_value, 2),
            "improvement_direction": direction,
            "risk_impact_if_improved_10_percent": {
                "mortality_30d_change": round((improved_pred - baseline_pred) * 100, 2),
                "new_risk": round(improved_pred, 3),
            },
            "risk_impact_if_worsened_10_percent": {
                "mortality_30d_change": round((worsened_pred - baseline_pred) * 100, 2),
                "new_risk": round(worsened_pred, 3),
            },
            "modifiable": feat_meta["modifiable"],
            "clinical_intervention": feat_meta["intervention"],
            "sensitivity_magnitude": abs(round((worsened_pred - improved_pred) * 100, 2)),
        })

    sensitivity_results.sort(key=lambda x: -x["sensitivity_magnitude"])
    return sensitivity_results[:6]


def analyze_cost_effectiveness(scenarios: List[Dict], patient_age: int) -> Dict:
    """
    Incremental cost-effectiveness analysis (ICEA) comparing treatment scenarios.
    Returns cost per QALY gained and cost-effectiveness acceptability.

    QALY model:
      - Remaining life expectancy discounted at 3% annually (standard health economics)
      - 30-day mortality risk directly reduces expected life years
      - Baseline QoL for acute illness = 0.65 (EQ-5D population norms for pneumonia)
      - Treatment QoL uplift based on recovery probability
      - Complications reduce QoL by 0.10 per event
    """
    if not scenarios:
        return {}

    DISCOUNT_RATE = 0.03
    life_expectancy_remaining = max(1, 85 - patient_age)

    # Discounted life years: sum(1/(1+r)^t) for t=1..N
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

        # Survival-adjusted discounted life years
        prob_surviving_30d = 1.0 - mortality_risk_30d
        survival_adjusted_ly = discounted_ly * prob_surviving_30d

        # QoL: baseline pneumonia QoL + recovery uplift - complication penalty
        baseline_qol   = 0.65
        recovery_uplift = recovery_prob_7d * 0.20   # max +0.20 if full recovery
        complication_penalty = complication_risk * 0.10
        treated_qol = min(0.95, max(0.10, baseline_qol + recovery_uplift - complication_penalty))

        qalys = round(treated_qol * survival_adjusted_ly, 2)

        # Cost: use provided cost, or impute transparently
        cost = scenario.get("estimated_cost_usd")
        cost_source = "provided"
        if cost is None or cost == 0:
            if "hospitalization" in [i.lower() for i in scenario.get("interventions", [])]:
                cost = 15_000
            elif scenario.get("option_id") == "C" and not scenario.get("drugs"):
                cost = 0   # true no-treatment baseline
            else:
                cost = 500
            cost_source = "imputed"

        cea_results.append({
            "option_id": scenario["option_id"],
            "label": scenario["label"],
            "estimated_cost_usd": cost,
            "cost_source": cost_source,
            "survival_probability_30d": round(prob_surviving_30d, 3),
            "quality_of_life_score": round(treated_qol, 3),
            "estimated_qalys": qalys,
            "cost_per_qaly": round(cost / max(qalys, 0.01), 2) if qalys > 0 else 999_999,
        })

    cea_results.sort(key=lambda x: x["cost_per_qaly"])

    for result in cea_results:
        cpq = result["cost_per_qaly"]
        if cpq < 100_000:
            result["cost_effective"] = True
        elif cpq < 150_000:
            result["cost_effective"] = "BORDERLINE"
        else:
            result["cost_effective"] = False

    return {
        "scenarios": cea_results,
        "most_cost_effective": cea_results[0]["option_id"] if cea_results else None,
        "threshold_used_usd_per_qaly": 100_000,
        "discount_rate_used": DISCOUNT_RATE,
        "interpretation": (
            "Incremental cost-effectiveness analysis. QALYs discounted at 3% annually. "
            "Survival-adjusted for 30-day mortality risk. "
            "Standard threshold: $100,000/QALY in US healthcare. "
            "Costs marked 'imputed' are estimated — provide actual costs for precise analysis."
        ),
    }


# ── LLM Narrative Builder ─────────────────────────────────────────────────────

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
    Generate evidence-based clinical narrative with enhanced reasoning.
    Falls back to rule-based narrative if LLM is unavailable.
    """
    demo = patient_state.get("demographics", {})
    age = demo.get("age", "?")
    gender = demo.get("gender", "patient")
    best = next((s for s in scenarios if s["option_id"] == recommended_option), scenarios[0] if scenarios else {})

    if not llm_ready or not llm:
        rec_label = best.get("label", recommended_option)
        rec_7d = best.get("predictions", {}).get("recovery_probability_7d", 0)
        rec_mort = best.get("predictions", {}).get("mortality_risk_30d", 0)

        narrative = (
            f"For this {age}-year-old {gender} with {diagnosis} ({risk_profile} risk), "
            f"{rec_label} offers {rec_7d:.0%} 7-day recovery probability and "
            f"{rec_mort:.0%} 30-day mortality risk. "
        )

        if sensitivity_top_3:
            top = sensitivity_top_3[0]
            narrative += (
                f"Most modifiable risk factor: {top['feature_name']} "
                f"(10% improvement reduces mortality by "
                f"{abs(top['risk_impact_if_improved_10_percent']['mortality_30d_change']):.1f}% points). "
            )

        if cost_effectiveness:
            most_ce = cost_effectiveness.get("most_cost_effective")
            if most_ce == recommended_option:
                narrative += "This option is also most cost-effective. "

        narrative += "Evidence-based decision support — clinical judgment required."
        return narrative

    # Build scenario lines for LLM
    scenario_lines = []
    for s in scenarios:
        preds = s.get("predictions", {})
        guideline_raw = s.get("guideline_adherence", {})

        # guideline_adherence can be a list (multi-drug) or single dict
        if isinstance(guideline_raw, list):
            # Use the highest-priority (first after sorting) result for the narrative label
            guideline = guideline_raw[0] if guideline_raw else {}
        else:
            guideline = guideline_raw if guideline_raw else {}

        adherence_note = ""
        if guideline:
            adherence = guideline.get("adherence", "")
            if adherence == "FIRST_LINE":
                adherence_note = " [FIRST-LINE per guidelines]"
            elif adherence == "SECOND_LINE":
                adherence_note = " [SECOND-LINE]"
            elif adherence == "OFF_GUIDELINE":
                adherence_note = " [OFF-GUIDELINE]"

        scenario_lines.append(
            f"Option {s['option_id']} ({s['label']}){adherence_note}: "
            f"7d recovery {preds.get('recovery_probability_7d', 0):.0%}, "
            f"30d mortality {preds.get('mortality_risk_30d', 0):.0%} "
            f"(95% CI: {preds.get('mortality_risk_30d_ci', '?')}), "
            f"30d readmission {preds.get('readmission_risk_30d', 0):.0%}"
        )

    conditions = patient_state.get("active_conditions", [])
    comorbidity_summary = ", ".join([c.get("display", "") for c in conditions[:3]])

    sensitivity_context = ""
    if sensitivity_top_3:
        sensitivity_context = "\n\nKey modifiable risk factors:\n" + "\n".join([
            f"- {s['feature_name']}: 10% improvement → "
            f"{abs(s['risk_impact_if_improved_10_percent']['mortality_30d_change']):.1f}% mortality reduction"
            for s in sensitivity_top_3[:3]
        ])

    cost_context = ""
    if cost_effectiveness:
        most_ce = cost_effectiveness.get("most_cost_effective")
        cost_context = (
            f"\n\nCost-effectiveness: Option {most_ce} is most cost-effective "
            "at current willingness-to-pay threshold."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical decision support system providing evidence-based treatment recommendations. "
         "Write exactly 4-5 sentences using specific data from the simulation. "
         "Focus on: (1) patient-specific risk factors, (2) numerical comparison of outcomes, "
         "(3) guideline adherence, (4) key modifiable factors. "
         "End with: 'This is AI-generated decision support requiring physician validation.'"),
        ("human",
         f"Patient: {age}y {gender}. "
         f"Diagnosis: {diagnosis} (ICD-10: {diagnosis_code or 'not specified'}). "
         f"Comorbidities: {comorbidity_summary or 'none documented'}. "
         f"Risk profile: {risk_profile}.\n\n"
         f"Treatment scenarios:\n" + "\n".join(scenario_lines) + "\n\n"
         f"Recommended: Option {recommended_option}"
         f"{sensitivity_context}"
         f"{cost_context}\n\n"
         "Provide evidence-based clinical narrative justifying the recommendation."),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        narrative = chain.invoke({})
        if "AI-generated" not in narrative:
            narrative += " This is AI-generated decision support requiring physician validation."
        return narrative
    except Exception as e:
        print(f"  ⚠️  LLM narrative failed: {e}")
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

    # Add monitoring from guidelines if available
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

    # Risk attribution extension
    risk_attribution_extension = [
        {
            "url": "http://meditwin.ai/fhir/StructureDefinition/risk-factor",
            "extension": [
                {"url": "factor",       "valueString":  attr["feature"]},
                {"url": "contribution", "valueString":  attr["contribution"]},
                {"url": "direction",    "valueString":  attr["direction"]},
                {"url": "importance",   "valueDecimal": attr.get("importance_score", 0)},
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
                    {"url": "recovery-probability-7d", "valueDecimal": predicted_recovery},
                    {"url": "confidence-interval",     "valueString":  "95% CI available in source"},
                ],
            },
        ] + risk_attribution_extension,
        "status": "active",
        "intent": "proposal",  # requires physician review before becoming an order
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