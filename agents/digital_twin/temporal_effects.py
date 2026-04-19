"""
agents/digital_twin/temporal_effects.py (NEW FILE - P3 FIX)
============================================================
Temporal treatment effect modeling for pneumonia treatment.

Tracks clinical response at key decision points:
- 24h: Early response (fever curve, clinical appearance)
- 72h: Clinical improvement threshold (48-72h rule)
- 7d: Recovery probability
- 30d: Readmission/mortality endpoints

Evidence base: CAP treatment guidelines, observational cohorts.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TemporalMilestone:
    """Single time point in treatment trajectory."""
    timepoint: str  # "24h", "72h", "7d", etc.
    clinical_improvement_probability: float
    escalation_triggers: List[str]
    monitoring_actions: List[str]
    expected_clinical_findings: Dict[str, str]


# ── Evidence-Based Treatment Response Timelines ───────────────────────────────

TREATMENT_RESPONSE_PROFILES = {
    # Azithromycin monotherapy (oral outpatient)
    "azithromycin_oral": {
        "24h": {
            "improvement_rate": 0.35,  # 35% show early improvement
            "triggers": [
                "Persistent fever >38.5°C",
                "Worsening dyspnea",
                "New confusion",
                "O2 sat <90% on room air",
            ],
            "monitoring": [
                "Vital signs q4h",
                "Symptom diary",
                "Phone follow-up at 24h",
            ],
            "expected": {
                "temperature": "May remain elevated (38-38.5°C) — normal at 24h",
                "respiratory_rate": "Unchanged or slightly improved",
                "energy": "Minimal change expected",
            },
        },
        "72h": {
            "improvement_rate": 0.78,  # 78% improved by 72h (48-72h rule)
            "triggers": [
                "No fever improvement (still >38°C)",
                "Worsening infiltrates on repeat CXR",
                "Rising WBC or CRP",
                "Patient reports worsening symptoms",
            ],
            "monitoring": [
                "Clinical reassessment",
                "Repeat WBC/CRP if initial critical",
                "Consider CXR if deteriorating",
            ],
            "expected": {
                "temperature": "Defervescence expected (<37.5°C)",
                "respiratory_rate": "Improved or normalizing",
                "energy": "Patient reports 'feeling better'",
            },
        },
        "7d": {
            "improvement_rate": 0.85,
            "triggers": [
                "Persistent symptoms at 7 days",
                "New complications",
            ],
            "monitoring": [
                "Phone follow-up",
                "Arrange follow-up CXR at 6 weeks",
            ],
            "expected": {
                "temperature": "Normal",
                "respiratory_rate": "Normal or near-normal",
                "energy": "Returning to baseline",
            },
        },
    },
    
    # Ceftriaxone IV + Azithromycin (inpatient)
    "ceftriaxone_azithromycin_iv": {
        "24h": {
            "improvement_rate": 0.55,  # Faster response with IV therapy
            "triggers": [
                "Hypotension (SBP <90)",
                "New oxygen requirement >4L",
                "Rising lactate",
                "Altered mental status",
            ],
            "monitoring": [
                "Continuous telemetry",
                "Vital signs q2h",
                "Strict I/O",
                "Serial lactate if initially elevated",
            ],
            "expected": {
                "temperature": "May spike at 12-24h (cytokine release), then improve",
                "respiratory_rate": "Slight improvement or stable",
                "hemodynamics": "Stable BP, improving if initially septic",
            },
        },
        "72h": {
            "improvement_rate": 0.88,  # Higher improvement with dual therapy
            "triggers": [
                "No clinical improvement",
                "Persistent bacteremia (if cultures positive)",
                "Worsening CXR",
            ],
            "monitoring": [
                "Repeat WBC/CRP",
                "Blood cultures if initially positive",
                "Consider CT chest if complicated",
            ],
            "expected": {
                "temperature": "Afebrile (<37.5°C)",
                "respiratory_rate": "Significantly improved",
                "oxygenation": "Weaning O2 or off O2",
            },
        },
        "7d": {
            "improvement_rate": 0.95,
            "triggers": [
                "Unable to complete PO challenge",
                "New complications (empyema, abscess)",
            ],
            "monitoring": [
                "Transition to oral antibiotics",
                "Discharge planning",
            ],
            "expected": {
                "temperature": "Normal",
                "respiratory_rate": "Normal",
                "activity": "Ambulating without dyspnea",
            },
        },
    },
}


# ── Complication Risk Over Time ───────────────────────────────────────────────

COMPLICATION_TRAJECTORIES = {
    "empyema_risk": {
        "peak_timing": "5-7 days",
        "early_predictors": ["persistent fever", "pleural effusion on imaging"],
        "late_predictors": ["persistent leukocytosis", "failure to improve by 72h"],
    },
    "treatment_failure": {
        "peak_timing": "48-72 hours",
        "early_predictors": ["rising WBC/CRP", "worsening hypoxia"],
        "late_predictors": ["no defervescence by 72h"],
    },
    "secondary_infection": {
        "peak_timing": "7-14 days (if hospitalized)",
        "early_predictors": ["ICU admission", "mechanical ventilation"],
        "late_predictors": ["new fever after improvement"],
    },
}


# ── Core Function ─────────────────────────────────────────────────────────────

def predict_temporal_trajectory(
    treatment_profile: str,
    baseline_risks: Dict[str, float],
    patient_age: int,
    comorbidity_count: int,
    critical_lab_count: int,
) -> Dict[str, TemporalMilestone]:
    """
    Generate temporal treatment trajectory with decision points.
    
    Args:
        treatment_profile: Key from TREATMENT_RESPONSE_PROFILES
        baseline_risks: Dict with mortality_30d, complication, readmission_30d
        patient_age: Patient age in years
        comorbidity_count: Number of active comorbidities
        critical_lab_count: Number of critical lab values
    
    Returns:
        Dict of TemporalMilestone objects keyed by timepoint
    """
    if treatment_profile not in TREATMENT_RESPONSE_PROFILES:
        # Fallback to azithromycin profile
        treatment_profile = "azithromycin_oral"
    
    profile = TREATMENT_RESPONSE_PROFILES[treatment_profile]
    milestones = {}
    
    # Adjust improvement rates based on patient risk factors
    age_penalty = max(0, (patient_age - 65) * 0.01)  # -1% per year over 65
    comorbidity_penalty = comorbidity_count * 0.02   # -2% per comorbidity
    critical_penalty = critical_lab_count * 0.05     # -5% per critical lab
    
    total_penalty = age_penalty + comorbidity_penalty + critical_penalty
    
    for timepoint, data in profile.items():
        base_improvement = data["improvement_rate"]
        adjusted_improvement = max(0.10, base_improvement - total_penalty)
        
        milestones[timepoint] = TemporalMilestone(
            timepoint=timepoint,
            clinical_improvement_probability=round(adjusted_improvement, 3),
            escalation_triggers=data["triggers"],
            monitoring_actions=data["monitoring"],
            expected_clinical_findings=data["expected"],
        )
    
    return milestones


def get_treatment_profile_key(drugs: List[str], interventions: List[str]) -> str:
    """
    Map treatment option to temporal response profile.
    
    Returns profile key for TREATMENT_RESPONSE_PROFILES.
    """
    drugs_lower = [d.lower() for d in drugs]
    interventions_lower = [i.lower() for i in interventions]
    
    # Ceftriaxone + Azithromycin IV (inpatient dual therapy)
    if any("ceftriaxone" in d for d in drugs_lower) and any("azithromycin" in d for d in drugs_lower):
        if any("hospitalization" in i or "iv" in i for i in interventions_lower):
            return "ceftriaxone_azithromycin_iv"
    
    # Azithromycin oral (outpatient monotherapy)
    if any("azithromycin" in d for d in drugs_lower):
        return "azithromycin_oral"
    
    # Default fallback
    return "azithromycin_oral"


def format_temporal_effects_for_response(
    milestones: Dict[str, TemporalMilestone],
    include_clinical_detail: bool = True,
) -> Dict:
    """
    Format temporal milestones for API response.
    
    Args:
        milestones: Dict of TemporalMilestone objects
        include_clinical_detail: Whether to include expected findings
    
    Returns:
        Formatted dict for JSON serialization
    """
    formatted = {}
    
    for timepoint, milestone in milestones.items():
        entry = {
            "timepoint": milestone.timepoint,
            "clinical_improvement_probability": milestone.clinical_improvement_probability,
            "escalation_triggers": milestone.escalation_triggers,
            "monitoring_actions": milestone.monitoring_actions,
        }
        
        if include_clinical_detail:
            entry["expected_clinical_findings"] = milestone.expected_clinical_findings
        
        formatted[timepoint] = entry
    
    return formatted


def get_failure_threshold_guidance(
    milestones: Dict[str, TemporalMilestone],
    baseline_mortality: float,
) -> Dict:
    """
    Generate treatment failure decision guidance.
    
    Returns dict with escalation criteria and timing.
    """
    # 72h is the critical decision point (48-72h rule in pneumonia)
    milestone_72h = milestones.get("72h")
    
    if not milestone_72h:
        return {}
    
    improvement_prob_72h = milestone_72h.clinical_improvement_probability
    
    # High-risk patients: stricter escalation criteria
    if baseline_mortality > 0.15:
        escalation_threshold = "No improvement at 48h"
        recommendation = "Consider early escalation given high baseline risk"
    elif baseline_mortality > 0.08:
        escalation_threshold = "No improvement at 72h"
        recommendation = "Standard 48-72h reassessment appropriate"
    else:
        escalation_threshold = "Worsening at 72h"
        recommendation = "Low risk — outpatient monitoring acceptable if stable"
    
    return {
        "critical_decision_point": "72h",
        "escalation_threshold": escalation_threshold,
        "expected_improvement_rate": improvement_prob_72h,
        "recommendation": recommendation,
        "escalation_options": [
            "Switch to broader-spectrum antibiotic (e.g., add vancomycin for MRSA)",
            "Add respiratory fluoroquinolone (levofloxacin/moxifloxacin)",
            "Hospital admission for IV therapy if outpatient",
            "Imaging to evaluate for complications (CT chest, ultrasound for effusion)",
        ],
    }


# ── Integration Helper ────────────────────────────────────────────────────────

def add_temporal_effects_to_scenario(
    scenario: Dict,
    baseline_risks: Dict[str, float],
    patient_age: int,
    comorbidity_count: int,
    critical_lab_count: int,
) -> Dict:
    """
    Add temporal effects to an existing scenario dict.
    Modifies scenario in-place and returns it.
    """
    profile_key = get_treatment_profile_key(
        scenario.get("drugs", []),
        scenario.get("interventions", []),
    )
    
    milestones = predict_temporal_trajectory(
        treatment_profile=profile_key,
        baseline_risks=baseline_risks,
        patient_age=patient_age,
        comorbidity_count=comorbidity_count,
        critical_lab_count=critical_lab_count,
    )
    
    scenario["temporal_predictions"] = format_temporal_effects_for_response(milestones)
    
    scenario["treatment_failure_guidance"] = get_failure_threshold_guidance(
        milestones,
        baseline_risks.get("mortality_30d", 0.1),
    )
    
    return scenario