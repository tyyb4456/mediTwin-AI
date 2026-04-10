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
    guideline = CLINICAL_GUIDELINES.get(diagnosis_code.split(".")[0])

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
def calculate_chadsvasc_score(
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

    # Features where lower is better (decrease by 10% to "improve")
    lower_is_better = {"wbc", "crp", "creatinine", "glucose", "critical_lab_count"}

    sensitivity_results = []

    for feat_name, feat_meta in modifiable_features.items():
        if feat_name not in feature_dict or feat_name not in feature_names:
            continue

        feat_idx = feature_names.index(feat_name)
        baseline_value = feature_vector[feat_idx]

        # 10% improvement
        improved_vector = feature_vector.copy()
        if feat_name in lower_is_better:
            improved_vector[feat_idx] = baseline_value * 0.9
        else:
            improved_vector[feat_idx] = baseline_value * 1.1
        improved_pred = model.predict_proba(np.array([improved_vector]))[0][1]

        # 10% worsening
        worsened_vector = feature_vector.copy()
        if feat_name in lower_is_better:
            worsened_vector[feat_idx] = baseline_value * 1.1
        else:
            worsened_vector[feat_idx] = baseline_value * 0.9
        worsened_pred = model.predict_proba(np.array([worsened_vector]))[0][1]

        sensitivity_results.append({
            "feature_name": feat_name,
            "baseline_value": round(baseline_value, 2),
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
    """
    if not scenarios:
        return {}

    life_expectancy_remaining = max(5, 85 - patient_age)
    cea_results = []

    for scenario in scenarios:
        preds = scenario.get("predictions", {})

        baseline_qol = 0.70
        recovery_prob = preds.get("recovery_probability_7d", 0.7)
        treated_qol = baseline_qol + (recovery_prob * 0.15)

        mortality_risk = preds.get("mortality_risk_30d", 0.05)
        survival_years = life_expectancy_remaining * (1 - mortality_risk)
        qalys = treated_qol * survival_years

        cost = scenario.get("estimated_cost_usd") or 0
        if cost == 0:
            cost = (
                15_000
                if "hospitalization" in [i.lower() for i in scenario.get("interventions", [])]
                else 500
            )

        cea_results.append({
            "option_id": scenario["option_id"],
            "label": scenario["label"],
            "estimated_cost_usd": cost,
            "estimated_qalys": round(qalys, 2),
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
        "interpretation": (
            "Cost-effectiveness analysis comparing total cost vs. quality-adjusted life years gained. "
            "Standard threshold: $100,000/QALY in US healthcare."
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
        guideline = s.get("guideline_adherence", {})

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