"""
agents/lab_analysis/lab_logic.py
----------------------------------
Shared business logic for the Lab Analysis Agent.
Imported by both main.py and stream_endpoint.py to avoid circular imports.

Contains:
  - Pydantic models (AlternativeDiagnosis, LabInterpretation, TrendAnalysis)
  - analyze_trends()
  - run_llm_interpretation()
  - build_llm_chain()          ← new: returns the raw structured chain for astream()
  - generate_clinical_decision_support()
  - _suggest_follow_up_labs()
  - build_lab_prompt_inputs()  ← new: builds (user_prompt) for reuse in streaming
"""

from __future__ import annotations

import logging
from typing import Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger("meditwin.lab_analysis.logic")


# ── Pydantic Models ───────────────────────────────────────────────────────────

class AlternativeDiagnosis(BaseModel):
    icd10_code: str = Field(description="ICD-10 code for alternative diagnosis")
    display: str = Field(description="Human-readable diagnosis name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this alternative")
    supporting_labs: List[str] = Field(description="Lab values supporting this diagnosis")


class LabInterpretation(BaseModel):
    pattern_interpretation: str = Field(
        description="2-3 sentence clinical narrative explaining the lab pattern"
    )
    confirms_top_diagnosis: bool = Field(
        description="Does the lab pattern support the proposed diagnosis?"
    )
    lab_confidence_boost: float = Field(
        ge=0.0, le=1.0,
        description="How much these labs add to diagnostic confidence (0.0-1.0)"
    )
    alternative_diagnoses: List[AlternativeDiagnosis] = Field(
        default_factory=list,
        description="Alternative diagnoses suggested by labs if they contradict primary"
    )
    reasoning: str = Field(
        description="Clinical reasoning for why labs confirm or contradict diagnosis"
    )
    clinical_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific clinical actions recommended based on lab findings"
    )
    urgency_level: str = Field(
        description="STAT | URGENT | ROUTINE",
        pattern="^(STAT|URGENT|ROUTINE)$"
    )


class TrendAnalysis(BaseModel):
    loinc: str
    display: str
    current_value: float
    previous_value: Optional[float] = None
    change_amount: Optional[float] = None
    change_percentage: Optional[float] = None
    trend_direction: str  # IMPROVING | WORSENING | STABLE | NEW
    clinical_significance: str


# ── Prompt ────────────────────────────────────────────────────────────────────

LAB_INTERPRETATION_SYSTEM = """You are an expert clinical pathologist providing structured laboratory interpretation.

CRITICAL RULES:
1. NEVER contradict CRITICAL flags from the rules engine — they are deterministic safety signals
2. Base ALL interpretations on the actual lab values provided
3. Be clinically precise and evidence-based
4. Recommend specific actionable next steps
5. Consider the patient's diagnosis context but remain objective

Your job is to:
- Interpret lab patterns in clinical context
- Confirm or question the proposed diagnosis based on objective lab evidence
- Suggest alternative diagnoses ONLY if labs strongly contradict the primary
- Recommend specific clinical actions
- Assign appropriate urgency level"""


# ── Prompt builder (shared between sync invoke and async astream) ─────────────

def build_llm_prompt(
    classified_labs: list[dict],
    patterns: list[dict],
    diagnosis_output: Optional[dict],
    critical_alerts: list[dict],
) -> tuple[ChatPromptTemplate, dict]:
    """
    Build the ChatPromptTemplate + input dict for LLM invocation.
    Returns (prompt, inputs) so both sync and async callers can reuse it.
    """
    abnormal = [
        f"{r['display']}: {r['value']} {r['unit']} [{r['flag']}] (ref: {r['reference_range']})"
        for r in classified_labs
        if r["flag"] in ("HIGH", "LOW", "CRITICAL")
    ]

    pattern_summary = "\n".join(
        f"- {p['pattern']}: {p['description']} (supports: {', '.join(p['supports_icd10'])})"
        for p in patterns
    ) if patterns else "No specific clinical patterns detected"

    critical_summary = "\n".join(
        f"⚠️ CRITICAL: {a['display']} {a['value']} — {a['message'][:100]}"
        for a in critical_alerts
    ) if critical_alerts else "No critical alerts"

    proposed_dx = diagnosis_output.get("top_diagnosis", "Unknown") if diagnosis_output else "Unknown"
    proposed_icd10 = diagnosis_output.get("top_icd10_code", "UNKNOWN") if diagnosis_output else "UNKNOWN"

    user_prompt = f"""ABNORMAL LAB RESULTS:
{chr(10).join(abnormal) if abnormal else "None"}

DETECTED CLINICAL PATTERNS:
{pattern_summary}

CRITICAL ALERTS:
{critical_summary}

PROPOSED DIAGNOSIS: {proposed_dx} (ICD-10: {proposed_icd10})

Provide structured clinical interpretation of these laboratory findings."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", LAB_INTERPRETATION_SYSTEM),
        ("human", user_prompt),
    ])

    return prompt, {}   # inputs dict is empty because user_prompt is baked in


def build_llm_chain(llm: ChatGoogleGenerativeAI):
    """Return the structured LLM — caller builds (prompt | chain) and can astream() it."""
    return llm.with_structured_output(LabInterpretation)


# ── Sync invocation (used by /analyze-labs non-streaming endpoint) ────────────

def run_llm_interpretation(
    classified_labs: list[dict],
    patterns: list[dict],
    diagnosis_output: Optional[dict],
    critical_alerts: list[dict],
    llm: Optional[ChatGoogleGenerativeAI],
    llm_ready: bool,
) -> Optional[LabInterpretation]:
    """
    Synchronous structured LLM interpretation.
    Returns LabInterpretation or None if LLM unavailable.
    """
    if not llm_ready or not llm:
        return None
    
    if not classified_labs:
        return LabInterpretation(
            pattern_interpretation="No laboratory data available.",
            confirms_top_diagnosis=False,
            lab_confidence_boost=0.0,
            reasoning="Cannot assess without labs",
            clinical_recommendations=[
                "Obtain baseline labs immediately"
            ],
            urgency_level="URGENT",
        )

    abnormal = [r for r in classified_labs if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    if not abnormal:
        return LabInterpretation(
            pattern_interpretation="All laboratory values within normal limits.",
            confirms_top_diagnosis=True,
            lab_confidence_boost=0.0,
            alternative_diagnoses=[],
            reasoning="Normal labs neither confirm nor contradict the proposed diagnosis.",
            clinical_recommendations=["Continue current monitoring", "Recheck if symptoms worsen"],
            urgency_level="ROUTINE",
        )

    prompt, inputs = build_llm_prompt(classified_labs, patterns, diagnosis_output, critical_alerts)

    try:
        structured_llm = build_llm_chain(llm)
        chain = prompt | structured_llm
        return chain.invoke(inputs)
    except Exception as e:
        logger.warning(f"Structured LLM interpretation failed: {e}")
        return None


# ── Trend Analysis ────────────────────────────────────────────────────────────

def analyze_trends(
    current_labs: List[dict],
    previous_labs: Optional[List[dict]],
    classified_current: List[dict],
) -> List[TrendAnalysis]:
    """Compare current labs to previous values to detect trends."""
    if not previous_labs:
        return []
    if not isinstance(previous_labs[0], dict):
        return []

    trends = []
    prev_by_loinc = {lab["loinc"]: lab for lab in previous_labs}

    for current in classified_current:
        loinc = current["loinc"]
        prev = prev_by_loinc.get(loinc)

        if not prev:
            trends.append(TrendAnalysis(
                loinc=loinc,
                display=current["display"],
                current_value=current["value"],
                trend_direction="NEW",
                clinical_significance=f"First measurement of {current['display']}",
            ))
            continue

        curr_val = current["value"]
        prev_val = prev.get("value")
        if prev_val is None:
            continue

        change = curr_val - prev_val
        pct_change = (change / prev_val * 100) if prev_val != 0 else 0

        direction = "STABLE"
        significance = ""

        if abs(pct_change) < 5:
            direction = "STABLE"
            significance = f"{current['display']} stable"
        else:
            curr_flag = current.get("flag", "NORMAL")
            prev_flag = prev.get("flag", "NORMAL")

            if change > 0:
                if curr_flag in ("HIGH", "CRITICAL") and prev_flag == "NORMAL":
                    direction = "WORSENING"
                    significance = f"{current['display']} newly elevated — rapid deterioration"
                elif curr_flag == "CRITICAL" and prev_flag == "HIGH":
                    direction = "WORSENING"
                    significance = f"{current['display']} critically elevated — urgent intervention"
                elif curr_flag == "NORMAL" and prev_flag in ("LOW", "CRITICAL"):
                    direction = "IMPROVING"
                    significance = f"{current['display']} normalizing — treatment response"
                else:
                    direction = "WORSENING"
                    significance = f"{current['display']} changed {pct_change:+.1f}%"
            else:
                if curr_flag in ("LOW", "CRITICAL") and prev_flag == "NORMAL":
                    direction = "WORSENING"
                    significance = f"{current['display']} newly decreased"
                elif curr_flag == "NORMAL" and prev_flag in ("HIGH", "CRITICAL"):
                    direction = "IMPROVING"
                    significance = f"{current['display']} normalizing — positive response"
                else:
                    direction = "IMPROVING" if curr_flag in ("HIGH", "CRITICAL") else "WORSENING"
                    significance = f"{current['display']} changed {pct_change:+.1f}%"

        trends.append(TrendAnalysis(
            loinc=loinc,
            display=current["display"],
            current_value=curr_val,
            previous_value=prev_val,
            change_amount=round(change, 2),
            change_percentage=round(pct_change, 1),
            trend_direction=direction,
            clinical_significance=significance,
        ))

    return trends

def sanitize_action(action: str) -> str:
    unsafe_verbs = ["initiate", "start", "administer", "give"]

    for verb in unsafe_verbs:
        if action.lower().startswith(verb):
            return action.replace(verb.capitalize(), "Recommend evaluation for", 1)

    return action

def map_priority(action: str) -> str:
    a = action.lower()

    if "immediate" in a or "stat" in a:
        return "STAT"
    if "antibiotic" in a or "sepsis" in a:
        return "URGENT"
    if "imaging" in a or "culture" in a:
        return "URGENT"
    if "glucose" in a or "monitor" in a:
        return "ROUTINE"

    return "ROUTINE"

# ── Clinical Decision Support ─────────────────────────────────────────────────

def generate_clinical_decision_support(
    classified_labs: List[dict],
    patterns: List[dict],
    critical_alerts: List[dict],
    llm_interpretation: Optional[LabInterpretation],
) -> dict:
    recommendations = []
    monitoring_plan = []
    consults_needed = []

    true_critical = [a for a in critical_alerts if a["level"] == "CRITICAL"]

    if true_critical:
        recommendations.append({
            "priority": "STAT",
            "action": "Immediate clinical review of critical lab values",
            "details": f"{len(true_critical)} critical value(s) require immediate attention",
            "timeframe": "Within 15 minutes",
        })

    for pattern in patterns:
        p = pattern["pattern"].lower()
        if "sepsis" in p:
            recommendations.append({
                "priority": "URGENT",
                "action": "Initiate sepsis workup",
                "details": "Blood cultures x2, lactate, consider broad-spectrum antibiotics",
                "timeframe": "Within 1 hour",
            })
            consults_needed.append("Infectious Disease")

        if "aki" in p or "kidney" in p:
            recommendations.append({
                "priority": "URGENT",
                "action": "Assess renal function and hydration status",
                "details": "Check urine output, review medications for nephrotoxins",
                "timeframe": "Within 2 hours",
            })
            monitoring_plan.append("Daily creatinine and electrolytes")
            consults_needed.append("Nephrology")

        if "cardiac" in p or "troponin" in p:
            recommendations.append({
                "priority": "STAT",
                "action": "Cardiology evaluation",
                "details": "Serial troponins q3h, EKG, consider cardiac catheterization",
                "timeframe": "Immediate",
            })
            consults_needed.append("Cardiology")

    if llm_interpretation and llm_interpretation.clinical_recommendations:
        for rec in llm_interpretation.clinical_recommendations:
            recommendations.append({
                "priority": map_priority(rec),
                "action": sanitize_action(rec),
                "details": "AI-generated recommendation based on lab pattern analysis",
                "timeframe": "Per urgency level",
            })

    return {
        "immediate_actions": [r for r in recommendations if r["priority"] == "STAT"],
        "urgent_actions": [r for r in recommendations if r["priority"] == "URGENT"],
        "routine_actions": [r for r in recommendations if r["priority"] == "ROUTINE"],
        "monitoring_plan": monitoring_plan or ["Routine lab monitoring per protocol"],
        "consultations_recommended": list(set(consults_needed)),
        "follow_up_labs": _suggest_follow_up_labs(classified_labs, patterns),
    }


def _suggest_follow_up_labs(classified_labs: List[dict], patterns: List[dict]) -> List[dict]:
    suggestions = []

    wbc_critical = any(
        r["loinc"] == "26464-8" and r["flag"] == "CRITICAL" for r in classified_labs
    )
    if wbc_critical:
        suggestions.append({
            "test": "Complete Blood Count with Differential",
            "loinc": "58410-2",
            "rationale": "Critical WBC requires full differential to assess cell types",
            "timing": "Immediate",
        })

    cr_high = any(
        r["loinc"] == "2160-0" and r["flag"] in ("HIGH", "CRITICAL") for r in classified_labs
    )
    if cr_high:
        suggestions.append({
            "test": "Blood Urea Nitrogen",
            "loinc": "3094-0",
            "rationale": "Assess BUN:Cr ratio for prerenal vs intrinsic AKI",
            "timing": "Within 6 hours",
        })

    crp_high = any(r["loinc"] == "1988-5" and r["flag"] in ("HIGH", "CRITICAL") for r in classified_labs)

    if crp_high:
        suggestions.append({
            "test": "Repeat CRP",
            "loinc": "1988-5",
            "rationale": "Monitor inflammatory response to treatment",
            "timing": "24-48 hours",
        })

    infection_pattern = any("infection" in p["pattern"].lower() for p in patterns)

    if infection_pattern:
        suggestions.append({
            "test": "Blood cultures",
            "loinc": "600-7",
            "rationale": "Identify causative organism before antibiotics",
            "timing": "Immediate",
        })

        suggestions.append({
            "test": "Serum Lactate",
            "loinc": "2524-7",
            "rationale": "Assess for sepsis and tissue hypoperfusion",
            "timing": "Immediate",
        })

    return suggestions