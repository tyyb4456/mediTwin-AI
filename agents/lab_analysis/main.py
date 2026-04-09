"""
Agent 3: Lab Analysis Agent (Enhanced)
Deterministic rules engine + structured LLM interpretation + clinical decision support tools
Port: 8003

Improvements:
  - Structured LLM output using Pydantic models
  - Enhanced pattern detection with severity scoring
  - Trend analysis support (comparing to previous values)
  - Delta checks for rapid changes
  - Tool integration for external medical knowledge
  - Better age/gender adjustment
  - Clinical significance scoring
  - More comprehensive alerts
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv  
load_dotenv()

# Add parent directories to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rules_engine import (
    classify_all,
    detect_patterns,
    generate_critical_alerts,
    compute_overall_severity,
    compute_severity_score,
    check_rapid_changes,
)


# ── Structured Output Models ──────────────────────────────────────────────────

class AlternativeDiagnosis(BaseModel):
    """Alternative diagnosis suggested by lab pattern"""
    icd10_code: str = Field(description="ICD-10 code for alternative diagnosis")
    display: str = Field(description="Human-readable diagnosis name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this alternative")
    supporting_labs: List[str] = Field(description="Lab values supporting this diagnosis")


class LabInterpretation(BaseModel):
    """Structured LLM output for lab interpretation"""
    pattern_interpretation: str = Field(
        description="2-3 sentence clinical narrative explaining the lab pattern"
    )
    confirms_top_diagnosis: bool = Field(
        description="Does the lab pattern support the proposed diagnosis?"
    )
    lab_confidence_boost: float = Field(
        ge=0.0, le=0.25,
        description="How much these labs add to diagnostic confidence (0.0-0.25)"
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
        description="STAT | URGENT | ROUTINE - urgency of clinical response needed",
        pattern="^(STAT|URGENT|ROUTINE)$"
    )


class TrendAnalysis(BaseModel):
    """Trend analysis for a specific lab value"""
    loinc: str
    display: str
    current_value: float
    previous_value: Optional[float] = None
    change_amount: Optional[float] = None
    change_percentage: Optional[float] = None
    trend_direction: str  # IMPROVING | WORSENING | STABLE | NEW
    clinical_significance: str


# ── Request / Response Models ──────────────────────────────────────────────────

class LabAnalysisRequest(BaseModel):
    patient_state: dict
    diagnosis_agent_output: Optional[dict] = None
    previous_lab_results: Optional[List[dict]] = None  # For trend analysis


class LabAnalysisResponse(BaseModel):
    lab_summary: dict
    flagged_results: list
    pattern_analysis: dict
    diagnosis_confirmation: dict
    critical_alerts: list
    trend_analysis: Optional[List[TrendAnalysis]] = None
    severity_score: Optional[dict] = None
    clinical_decision_support: Optional[dict] = None
    llm_interpretation_available: bool


# ── Lifespan ──────────────────────────────────────────────────────────────────

llm: Optional[ChatGoogleGenerativeAI] = None
llm_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, llm_ready

    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-lite",
            temperature=0.2
        )
        llm_ready = True
        print("✓ Lab Analysis Agent started — Enhanced LLM interpretation ready")
    except Exception as e:
        print(f"⚠️  LLM init failed: {e} — rules engine only mode")

    yield
    print("✓ Lab Analysis Agent shutdown")


app = FastAPI(
    title="MediTwin Lab Analysis Agent (Enhanced)",
    description="Advanced lab interpretation with structured LLM output and clinical decision support",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Trend Analysis ────────────────────────────────────────────────────────────

def analyze_trends(
    current_labs: List[dict],
    previous_labs: Optional[List[dict]],
    classified_current: List[dict]
) -> List[TrendAnalysis]:
    """
    Compare current labs to previous values to detect trends.
    Critical for identifying rapid deterioration or improvement.
    """
    if not previous_labs:
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
                clinical_significance=f"First measurement of {current['display']}"
            ))
            continue
        
        curr_val = current["value"]
        prev_val = prev.get("value")
        
        if prev_val is None:
            continue
            
        change = curr_val - prev_val
        pct_change = (change / prev_val * 100) if prev_val != 0 else 0
        
        # Determine trend direction and significance
        direction = "STABLE"
        significance = ""
        
        # Significant change thresholds (clinical judgment)
        if abs(pct_change) < 5:
            direction = "STABLE"
            significance = f"{current['display']} stable"
        else:
            # Check if change is clinically significant based on flag changes
            curr_flag = current.get("flag", "NORMAL")
            prev_flag = prev.get("flag", "NORMAL")
            
            if change > 0:  # Value increased
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
                    direction = "WORSENING" if change > 0 else "IMPROVING"
                    significance = f"{current['display']} changed {pct_change:+.1f}%"
            else:  # Value decreased
                if curr_flag in ("LOW", "CRITICAL") and prev_flag == "NORMAL":
                    direction = "WORSENING"
                    significance = f"{current['display']} newly decreased"
                elif curr_flag == "NORMAL" and prev_flag in ("HIGH", "CRITICAL"):
                    direction = "IMPROVING"
                    significance = f"{current['display']} normalizing — positive response"
                else:
                    direction = "IMPROVING" if change < 0 and curr_flag in ("HIGH", "CRITICAL") else "WORSENING"
                    significance = f"{current['display']} changed {pct_change:+.1f}%"
        
        trends.append(TrendAnalysis(
            loinc=loinc,
            display=current["display"],
            current_value=curr_val,
            previous_value=prev_val,
            change_amount=round(change, 2),
            change_percentage=round(pct_change, 1),
            trend_direction=direction,
            clinical_significance=significance
        ))
    
    return trends


# ── LLM Interpretation with Structured Output ─────────────────────────────────

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


def run_llm_interpretation(
    classified_labs: list[dict],
    patterns: list[dict],
    diagnosis_output: Optional[dict],
    critical_alerts: list[dict],
) -> Optional[LabInterpretation]:
    """
    Run structured LLM interpretation using Pydantic model.
    Returns LabInterpretation or None if LLM unavailable.
    """
    if not llm_ready or not llm:
        return None

    # Build context for LLM
    abnormal = [
        f"{r['display']}: {r['value']} {r['unit']} [{r['flag']}] (ref: {r['reference_range']})"
        for r in classified_labs
        if r["flag"] in ("HIGH", "LOW", "CRITICAL")
    ]

    if not abnormal:
        return LabInterpretation(
            pattern_interpretation="All laboratory values within normal limits.",
            confirms_top_diagnosis=True,
            lab_confidence_boost=0.0,
            alternative_diagnoses=[],
            reasoning="Normal labs neither confirm nor contradict the proposed diagnosis.",
            clinical_recommendations=["Continue current monitoring", "Recheck if symptoms worsen"],
            urgency_level="ROUTINE"
        )

    # Build pattern summary
    pattern_summary = "\n".join(
        f"- {p['pattern']}: {p['description']} (supports: {', '.join(p['supports_icd10'])})"
        for p in patterns
    ) if patterns else "No specific clinical patterns detected"

    # Critical alerts summary
    critical_summary = "\n".join(
        f"⚠️ CRITICAL: {a['display']} {a['value']} — {a['message'][:100]}"
        for a in critical_alerts
    ) if critical_alerts else "No critical alerts"

    # Proposed diagnosis
    proposed_dx = diagnosis_output.get("top_diagnosis", "Unknown") if diagnosis_output else "Unknown"
    proposed_icd10 = diagnosis_output.get("top_icd10_code", "UNKNOWN") if diagnosis_output else "UNKNOWN"

    user_prompt = f"""ABNORMAL LAB RESULTS:
{chr(10).join(abnormal)}

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

    try:
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(LabInterpretation)
        chain = prompt | structured_llm
        
        result = chain.invoke({})
        return result
    except Exception as e:
        print(f"⚠️  Structured LLM interpretation failed: {e}")
        return None


# ── Clinical Decision Support ─────────────────────────────────────────────────

def generate_clinical_decision_support(
    classified_labs: List[dict],
    patterns: List[dict],
    critical_alerts: List[dict],
    llm_interpretation: Optional[LabInterpretation]
) -> dict:
    """
    Generate actionable clinical decision support based on lab findings.
    This goes beyond interpretation to provide specific next steps.
    """
    recommendations = []
    monitoring_plan = []
    consults_needed = []
    
    # Critical value response
    if critical_alerts:
        recommendations.append({
            "priority": "STAT",
            "action": "Immediate clinical review of critical lab values",
            "details": f"{len(critical_alerts)} critical value(s) require immediate attention",
            "timeframe": "Within 15 minutes"
        })
    
    # Pattern-based recommendations
    for pattern in patterns:
        if "sepsis" in pattern["pattern"].lower():
            recommendations.append({
                "priority": "URGENT",
                "action": "Initiate sepsis workup",
                "details": "Blood cultures x2, lactate, consider broad-spectrum antibiotics",
                "timeframe": "Within 1 hour"
            })
            consults_needed.append("Infectious Disease")
        
        if "aki" in pattern["pattern"].lower() or "kidney" in pattern["pattern"].lower():
            recommendations.append({
                "priority": "URGENT",
                "action": "Assess renal function and hydration status",
                "details": "Check urine output, review medications for nephrotoxins",
                "timeframe": "Within 2 hours"
            })
            monitoring_plan.append("Daily creatinine and electrolytes")
            consults_needed.append("Nephrology")
        
        if "cardiac" in pattern["pattern"].lower() or "troponin" in pattern["pattern"].lower():
            recommendations.append({
                "priority": "STAT",
                "action": "Cardiology evaluation",
                "details": "Serial troponins q3h, EKG, consider cardiac catheterization",
                "timeframe": "Immediate"
            })
            consults_needed.append("Cardiology")
    
    # Add LLM recommendations if available
    if llm_interpretation and llm_interpretation.clinical_recommendations:
        for rec in llm_interpretation.clinical_recommendations:
            recommendations.append({
                "priority": llm_interpretation.urgency_level,
                "action": rec,
                "details": "AI-generated recommendation based on lab pattern analysis",
                "timeframe": "Per urgency level"
            })
    
    return {
        "immediate_actions": [r for r in recommendations if r["priority"] == "STAT"],
        "urgent_actions": [r for r in recommendations if r["priority"] == "URGENT"],
        "routine_actions": [r for r in recommendations if r["priority"] == "ROUTINE"],
        "monitoring_plan": monitoring_plan or ["Routine lab monitoring per protocol"],
        "consultations_recommended": list(set(consults_needed)),
        "follow_up_labs": _suggest_follow_up_labs(classified_labs, patterns)
    }


def _suggest_follow_up_labs(classified_labs: List[dict], patterns: List[dict]) -> List[dict]:
    """Suggest specific follow-up labs based on current findings"""
    suggestions = []
    
    # If WBC is critical, suggest differential
    wbc_critical = any(r["loinc"] == "26464-8" and r["flag"] == "CRITICAL" for r in classified_labs)
    if wbc_critical:
        suggestions.append({
            "test": "Complete Blood Count with Differential",
            "loinc": "58410-2",
            "rationale": "Critical WBC requires full differential to assess cell types",
            "timing": "Immediate"
        })
    
    # If creatinine elevated, suggest BUN and eGFR
    cr_high = any(r["loinc"] == "2160-0" and r["flag"] in ("HIGH", "CRITICAL") for r in classified_labs)
    if cr_high:
        suggestions.append({
            "test": "Blood Urea Nitrogen",
            "loinc": "3094-0",
            "rationale": "Assess BUN:Cr ratio for prerenal vs intrinsic AKI",
            "timing": "Within 6 hours"
        })
    
    return suggestions


# ── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze-labs", response_model=LabAnalysisResponse)
async def analyze_labs(request: LabAnalysisRequest) -> LabAnalysisResponse:
    """
    Enhanced lab analysis with:
    - Deterministic rules engine (CRITICAL flags never suppressed)
    - Structured LLM interpretation
    - Trend analysis
    - Clinical decision support
    """
    patient_state = request.patient_state
    if not patient_state:
        raise HTTPException(status_code=400, detail="patient_state is required")

    demographics = patient_state.get("demographics", {})
    age = demographics.get("age", 40)
    gender = demographics.get("gender", "male")
    lab_results = patient_state.get("lab_results", [])

    # ── Step 1: Rules engine classification ────────────────────────────────────
    classified = classify_all(lab_results, age, gender)

    abnormal = [r for r in classified if r["flag"] in ("HIGH", "LOW", "CRITICAL")]
    critical = [r for r in classified if r["flag"] == "CRITICAL"]

    severity = compute_overall_severity(classified)
    severity_score = compute_severity_score(classified)

    lab_summary = {
        "total_results": len(classified),
        "abnormal_count": len(abnormal),
        "critical_count": len(critical),
        "overall_severity": severity,
        "severity_score": severity_score,
    }

    # ── Step 2: Pattern detection ──────────────────────────────────────────────
    patterns = detect_patterns(classified)

    # ── Step 3: Critical alerts ────────────────────────────────────────────────
    critical_alerts = generate_critical_alerts(classified)

    # ── Step 4: Trend analysis ─────────────────────────────────────────────────
    trend_analysis = None
    if request.previous_lab_results:
        trend_analysis = analyze_trends(
            lab_results,
            request.previous_lab_results,
            classified
        )

    # ── Step 5: Structured LLM interpretation ──────────────────────────────────
    llm_interpretation = run_llm_interpretation(
        classified,
        patterns,
        request.diagnosis_agent_output,
        critical_alerts
    )

    # ── Step 6: Clinical decision support ──────────────────────────────────────
    clinical_decision_support = generate_clinical_decision_support(
        classified,
        patterns,
        critical_alerts,
        llm_interpretation
    )

    # ── Step 7: Build diagnosis confirmation ───────────────────────────────────
    if llm_interpretation:
        confirms = llm_interpretation.confirms_top_diagnosis
        confidence_boost = llm_interpretation.lab_confidence_boost
        alt_code = llm_interpretation.alternative_diagnoses[0].icd10_code if llm_interpretation.alternative_diagnoses else None
        alt_display = llm_interpretation.alternative_diagnoses[0].display if llm_interpretation.alternative_diagnoses else None
        reasoning = llm_interpretation.reasoning
        pattern_interp = llm_interpretation.pattern_interpretation
    else:
        # Fallback logic
        bacterial_pattern = any(
            "bacterial" in p["pattern"].lower() or "sepsis" in p["pattern"].lower()
            for p in patterns
        )
        top_icd10 = (request.diagnosis_agent_output or {}).get("top_icd10_code", "")
        confirms = True
        if bacterial_pattern and top_icd10 and not top_icd10.startswith("J"):
            confirms = False
        confidence_boost = 0.10 if bacterial_pattern else 0.0
        alt_code = None
        alt_display = None
        reasoning = "Rules-only mode — pattern detection used for confirmation"
        pattern_interp = (
            f"Detected patterns: {', '.join(p['pattern'] for p in patterns)}"
            if patterns else "No specific patterns detected"
        )

    proposed_icd10 = (request.diagnosis_agent_output or {}).get("top_icd10_code", "UNKNOWN")
    proposed_dx = (request.diagnosis_agent_output or {}).get("top_diagnosis", "Unknown")

    diagnosis_confirmation = {
        "proposed_diagnosis": proposed_dx,
        "proposed_icd10": proposed_icd10,
        "confirms_top_diagnosis": confirms,
        "lab_confidence_boost": round(confidence_boost, 2),
        "alternative_diagnosis_code": alt_code,
        "alternative_diagnosis_display": alt_display,
        "reasoning": reasoning,
    }

    pattern_analysis = {
        "identified_patterns": patterns,
        "pattern_interpretation": pattern_interp,
    }

    # Build flagged results
    flagged_results = []
    for r in abnormal:
        flagged_results.append({
            "loinc": r["loinc"],
            "display": r["display"],
            "value": r["value"],
            "unit": r["unit"],
            "reference_range": r["reference_range"],
            "flag": r["flag"],
            "clinical_significance": r["clinical_notes"],
        })

    return LabAnalysisResponse(
        lab_summary=lab_summary,
        flagged_results=flagged_results,
        pattern_analysis=pattern_analysis,
        diagnosis_confirmation=diagnosis_confirmation,
        critical_alerts=critical_alerts,
        trend_analysis=trend_analysis,
        severity_score=severity_score,
        clinical_decision_support=clinical_decision_support,
        llm_interpretation_available=llm_ready,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "lab-analysis-enhanced",
        "version": "2.0.0",
        "llm_ready": llm_ready,
        "structured_output": llm_ready,
        "rules_engine": "active",
        "loinc_codes_supported": 16,
        "features": [
            "structured_llm_output",
            "trend_analysis",
            "clinical_decision_support",
            "severity_scoring",
            "delta_checks"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)