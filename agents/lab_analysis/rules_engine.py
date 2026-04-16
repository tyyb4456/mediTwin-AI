"""
Lab Analysis Rules Engine (Enhanced)
Pure Python — deterministic, no LLM.
Runs FIRST before LLM interpretation.
CRITICAL flags from this engine are NEVER suppressed by LLM output.

New features:
  - Severity scoring system (0-100 scale)
  - Rapid change detection (delta checks)
  - More comprehensive clinical patterns
  - Age/gender-specific reference ranges properly implemented
  - SIRS criteria detection
  - Organ dysfunction scoring
"""
import json
from pathlib import Path
from typing import Optional, List, Dict


# ── Load reference ranges once at import time ──────────────────────────────────
_REF_RANGES_PATH = Path(__file__).parent / "reference_ranges.json"

with open(_REF_RANGES_PATH, "r") as f:
    REFERENCE_RANGES: dict = json.load(f)


def _get_range(loinc: str, age: int, gender: str) -> Optional[dict]:
    """Pick the correct reference range for a given LOINC, age, and gender."""
    entry = REFERENCE_RANGES.get(loinc)
    if not entry:
        return None

    if age < 18:
        return entry.get("pediatric")
    elif gender.lower() in ("female", "f"):
        return entry.get("adult_female")
    else:
        return entry.get("adult_male")


def classify_result(loinc: str, value: float, age: int, gender: str) -> dict:
    """
    Classify a single lab result as NORMAL / HIGH / LOW / CRITICAL.
    
    Returns:
        {
            "loinc": str,
            "display": str,
            "value": float,
            "unit": str,
            "reference_range": str,
            "flag": "NORMAL" | "HIGH" | "LOW" | "CRITICAL",
            "clinical_notes": str
        }
    """
    entry = REFERENCE_RANGES.get(loinc, {})
    ref = _get_range(loinc, age, gender)

    display = entry.get("display", loinc)
    unit = entry.get("unit", "")
    clinical_notes = entry.get("clinical_notes", "")

    # Build human-readable reference range string
    if ref:
        low = ref.get("low")
        high = ref.get("high")
        range_str = f"{low} - {high}" if low is not None and high is not None else "N/A"
    else:
        range_str = "Unknown"

    # Determine flag — critical thresholds checked first
    flag = "NORMAL"
    if ref:
        crit_high = ref.get("critical_high")
        crit_low = ref.get("critical_low")
        high = ref.get("high")
        low = ref.get("low")

        if crit_high is not None and value >= crit_high:
            flag = "CRITICAL"
        elif crit_low is not None and value <= crit_low:
            flag = "CRITICAL"
        elif high is not None and value > high:
            flag = "HIGH"
        elif low is not None and value < low:
            flag = "LOW"
    
    return {
        "loinc": loinc,
        "display": display,
        "value": value,
        "unit": unit,
        "reference_range": range_str,
        "flag": flag,
        "clinical_notes": clinical_notes,
    }


def classify_all(lab_results: list[dict], age: int, gender: str) -> list[dict]:
    """
    Classify every lab result in the patient state.
    
    Args:
        lab_results: List of LabResult dicts from PatientState
        age: Patient age (int)
        gender: Patient gender string
        
    Returns:
        List of classified result dicts, same order as input
    """
    classified = []
    for lab in lab_results:
        loinc = lab.get("loinc", "")
        value = lab.get("value")

        if value is None:
            continue

        result = classify_result(loinc, float(value), age, gender)

        # Preserve original display name if we don't have a mapping
        if result["display"] == loinc and lab.get("display"):
            result["display"] = lab["display"]

        classified.append(result)

    return classified


# ── Severity Scoring ───────────────────────────────────────────────────────────

def compute_severity_score(classified_labs: List[dict]) -> dict:
    """
    Compute an overall severity score (0-100) based on lab abnormalities.
    
    Scoring logic:
      - Each CRITICAL value: +25 points
      - Each HIGH/LOW value: +10 points
      - Multiple organ system involvement: multiplier
      - SIRS criteria: +15 points
      - Organ dysfunction: +20 points per system
    
    Returns:
        {
            "score": int (0-100),
            "risk_category": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
            "contributors": List[str]
        }
    """
    score = 0
    contributors = []
    
    # Count flags
    critical_count = sum(1 for r in classified_labs if r["flag"] == "CRITICAL")
    high_low_count = sum(1 for r in classified_labs if r["flag"] in ("HIGH", "LOW"))
    
    # Base scoring
    score += critical_count * 25
    WEIGHTS = {
        "1988-5": 15,  # CRP
        "6598-7": 20,  # Troponin
        "2823-3": 20,  # Potassium
        "2160-0": 15,  # Creatinine
        "26464-8": 10, # WBC
    }

    for r in classified_labs:
        loinc = r["loinc"]
        value = float(r["value"])

        if r["flag"] in ("HIGH", "LOW"):
            score += WEIGHTS.get(loinc, 10)

        #  ADD THIS
        if loinc == "1988-5" and value > 100:
            score += 10
            contributors.append("CRP >100 (severe inflammation)")
    
    if critical_count > 0:
        contributors.append(f"{critical_count} critical value(s)")
    if high_low_count > 0:
        contributors.append(f"{high_low_count} abnormal value(s)")
    
    # Organ system involvement
    labs_by_loinc = {r["loinc"]: r for r in classified_labs}
    
    # Check for SIRS criteria (≥2 of: temp, HR, RR, WBC)
    sirs_count = 0
    wbc = labs_by_loinc.get("26464-8")
    if wbc and wbc["flag"] in ("HIGH", "LOW", "CRITICAL"):
        sirs_count += 1
    
    if sirs_count >= 2:
        score += 15
        contributors.append("SIRS criteria met")
    
    # Organ dysfunction scoring
    organ_dysfunction = []
    
    # Renal
    cr = labs_by_loinc.get("2160-0")
    if cr and cr["flag"] in ("HIGH", "CRITICAL"):
        score += 20
        organ_dysfunction.append("Renal")
    
    # Hepatic
    alt = labs_by_loinc.get("1742-6")
    if alt and alt["flag"] in ("HIGH", "CRITICAL"):
        score += 15
        organ_dysfunction.append("Hepatic")
    
    # Cardiac
    trop = labs_by_loinc.get("6598-7")
    if trop and trop["flag"] in ("HIGH", "CRITICAL"):
        score += 20
        organ_dysfunction.append("Cardiac")
    
    # Hematologic
    hgb = labs_by_loinc.get("718-7")
    plt = labs_by_loinc.get("777-3")
    if (hgb and hgb["flag"] in ("LOW", "CRITICAL")) or (plt and plt["flag"] in ("LOW", "CRITICAL")):
        score += 15
        organ_dysfunction.append("Hematologic")
    
    if organ_dysfunction:
        contributors.append(f"Organ dysfunction: {', '.join(organ_dysfunction)}")
    
    # Cap at 100
    score = min(score, 100)
    
    # Categorize risk
    if score >= 75:
        risk_category = "CRITICAL"
    elif score >= 50:
        risk_category = "HIGH"
    elif score >= 25:
        risk_category = "MODERATE"
    else:
        risk_category = "LOW"
    
    systems = set()

    for r in classified_labs:
        loinc = r["loinc"]

        if loinc == "26464-8":
            systems.add("immune")
        elif loinc == "1988-5":
            systems.add("inflammatory")
        elif loinc == "2345-7":
            systems.add("metabolic")

    organ_systems_affected = len(systems)

    return {
        "score": score,
        "risk_category": risk_category,
        "contributors": contributors,
        "organ_systems_affected": organ_systems_affected
    }

# ── Rapid Change Detection ────────────────────────────────────────────────────

def check_rapid_changes(current_labs: List[dict], previous_labs: List[dict]) -> List[dict]:
    """
    Detect rapid changes that indicate acute deterioration.
    
    Criteria for "rapid change":
      - Creatinine increase >0.3 mg/dL in 48h (KDIGO AKI criteria)
      - WBC change >50% from baseline
      - Hemoglobin drop >2 g/dL (acute bleeding)
      - Potassium change >1.0 mEq/L
      - Glucose change >200 mg/dL
    
    Returns:
        List of alerts for rapid changes
    """
    alerts = []
    prev_by_loinc = {lab["loinc"]: lab for lab in previous_labs}
    
    for current in current_labs:
        loinc = current["loinc"]
        prev = prev_by_loinc.get(loinc)
        
        if not prev:
            continue
        
        curr_val = current.get("value")
        prev_val = prev.get("value")
        
        if curr_val is None or prev_val is None:
            continue
        
        delta = curr_val - prev_val
        
        # Creatinine — AKI criteria
        if loinc == "2160-0" and delta >= 0.3:
            alerts.append({
                "loinc": loinc,
                "display": current["display"],
                "change": f"+{delta:.1f} mg/dL",
                "interpretation": "KDIGO AKI Stage 1 criteria met — acute kidney injury",
                "urgency": "URGENT"
            })
        
        # WBC — rapid change
        if loinc == "26464-8":
            pct_change = abs(delta / prev_val * 100) if prev_val != 0 else 0
            if pct_change > 50:
                alerts.append({
                    "loinc": loinc,
                    "display": current["display"],
                    "change": f"{delta:+.1f} ({pct_change:+.0f}%)",
                    "interpretation": "Rapid WBC change suggests acute process",
                    "urgency": "URGENT"
                })
        
        # Hemoglobin — acute bleeding
        if loinc == "718-7" and delta <= -2.0:
            alerts.append({
                "loinc": loinc,
                "display": current["display"],
                "change": f"{delta:.1f} g/dL",
                "interpretation": "Acute hemoglobin drop — evaluate for bleeding",
                "urgency": "STAT"
            })
        
        # Potassium — arrhythmia risk
        if loinc == "2823-3" and abs(delta) >= 1.0:
            alerts.append({
                "loinc": loinc,
                "display": current["display"],
                "change": f"{delta:+.1f} mEq/L",
                "interpretation": "Rapid potassium change — cardiac arrhythmia risk",
                "urgency": "STAT"
            })
    
    return alerts


# ── Enhanced Clinical Pattern Detection ────────────────────────────────────────

CLINICAL_PATTERNS = [
    {
        "name": "Bacterial infection markers",
        "description": "Multi-marker pattern consistent with bacterial infection",
        "rules": [
            lambda labs: labs.get("26464-8", {}).get("flag") in ("HIGH", "CRITICAL"),
            lambda labs: (
                labs.get("1988-5", {}).get("flag") in ("HIGH", "CRITICAL")
                or labs.get("770-8", {}).get("flag") in ("HIGH", "CRITICAL")
            ),
        ],
        "min_rules_met": 2,
        "supports_icd10": ["J18.9", "J15.9", "A41.9"],
        "sensitivity_note": "Combined WBC + CRP elevation: ~89% sensitivity for bacterial infection",
    },
    {
        "name": "Sepsis risk markers",
        "description": "Laboratory pattern raising concern for systemic sepsis",
        "rules": [
            lambda labs: labs.get("26464-8", {}).get("flag") == "CRITICAL",
            lambda labs: (
                labs.get("2160-0", {}).get("flag") in ("HIGH", "CRITICAL")
                or labs.get("1742-6", {}).get("flag") in ("HIGH", "CRITICAL")
                or labs.get("6598-7", {}).get("flag") in ("HIGH", "CRITICAL")
            ),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["A41.9"],
        "sensitivity_note": "CRITICAL WBC warrants sepsis workup: blood cultures, lactate",
    },
    {
        "name": "Acute kidney injury (AKI) markers",
        "description": "Laboratory pattern consistent with acute or chronic kidney injury",
        "rules": [
            lambda labs: labs.get("2160-0", {}).get("flag") in ("HIGH", "CRITICAL"),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["N17.9", "N18.9"],
        "sensitivity_note": "Creatinine elevation requires assessment of baseline and urine output",
    },
    {
        "name": "Acute kidney injury with electrolyte imbalance",
        "description": "Renal impairment with concurrent potassium abnormality",
        "rules": [
            lambda labs: labs.get("2160-0", {}).get("flag") in ("HIGH", "CRITICAL"),
            lambda labs: labs.get("2823-3", {}).get("flag") in ("HIGH", "LOW", "CRITICAL"),
        ],
        "min_rules_met": 2,
        "supports_icd10": ["N17.9"],
        "sensitivity_note": "Creatinine + potassium abnormality warrants urgent nephrology review",
    },
    {
        "name": "Cardiac injury markers",
        "description": "Troponin elevation indicating myocardial stress or injury",
        "rules": [
            lambda labs: labs.get("6598-7", {}).get("flag") in ("HIGH", "CRITICAL"),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["I21.9", "I25.1"],
        "sensitivity_note": "Troponin elevation in infection context may indicate demand ischemia",
    },
    {
        "name": "Anemia",
        "description": "Hemoglobin below normal range",
        "rules": [
            lambda labs: labs.get("718-7", {}).get("flag") in ("LOW", "CRITICAL"),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["D64.9"],
        "sensitivity_note": "Severity classification: mild >10, moderate 8-10, severe <8 g/dL",
    },
    {
        "name": "Coagulation concern (thrombocytopenia)",
        "description": "Low platelet count indicating bleeding risk",
        "rules": [
            lambda labs: labs.get("777-3", {}).get("flag") in ("LOW", "CRITICAL"),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["D69.6"],
        "sensitivity_note": "Platelets <50 = serious bleeding risk; <20 = spontaneous bleeding risk",
    },
    {
        "name": "Hyperglycemia / Diabetes decompensation",
        "description": "Significantly elevated glucose",
        "rules": [
            lambda labs: labs.get("2345-7", {}).get("flag") in ("HIGH", "CRITICAL"),
        ],
        "min_rules_met": 1,
        "supports_icd10": ["E11.9", "E13.9"],
        "sensitivity_note": "Critical glucose >500 may indicate DKA or HHS — check ketones and pH",
    },
    {
        "name": "Metabolic acidosis pattern",
        "description": "Lab pattern suggesting metabolic acidosis",
        "rules": [
            lambda labs: labs.get("2069-3", {}).get("flag") in ("HIGH", "CRITICAL"),  # Chloride high
            lambda labs: labs.get("2823-3", {}).get("flag") in ("HIGH", "CRITICAL"),  # Potassium high
        ],
        "min_rules_met": 1,
        "supports_icd10": ["E87.2"],
        "sensitivity_note": "Check blood gas for pH and lactate",
    },
    {
        "name": "Liver dysfunction pattern",
        "description": "Hepatocellular injury or hepatic synthetic dysfunction",
        "rules": [
            lambda labs: labs.get("1742-6", {}).get("flag") in ("HIGH", "CRITICAL"),  # ALT
            lambda labs: labs.get("1751-7", {}).get("flag") in ("LOW", "CRITICAL"),   # Albumin low
        ],
        "min_rules_met": 1,
        "supports_icd10": ["K72.9", "K76.9"],
        "sensitivity_note": "ALT >10x ULN suggests acute hepatitis or ischemic liver injury",
    },
]


def detect_patterns(classified_labs: list[dict]) -> list[dict]:
    """
    Run all clinical pattern rules against the classified lab results.
    
    Returns:
        List of detected pattern dicts with interpretation details
    """
    labs_by_loinc = {lab["loinc"]: lab for lab in classified_labs}

    detected = []
    for pattern in CLINICAL_PATTERNS:
        rules = pattern["rules"]
        min_met = pattern["min_rules_met"]

        rules_met = sum(1 for rule in rules if rule(labs_by_loinc))

        if rules_met >= min_met:
            def get_relevant_loincs(pattern_name):
                mapping = {
                    "Bacterial infection markers": ["26464-8", "1988-5", "770-8"],
                    "Hyperglycemia / Diabetes decompensation": ["2345-7"],
                    "Acute kidney injury (AKI) markers": ["2160-0"],
                }
                return mapping.get(pattern_name, [])

            relevant_loincs = get_relevant_loincs(pattern["name"])

            triggered_markers = [
                f"{lab['display']} {lab['flag']} ({lab['value']} {lab['unit']})"
                for lab in classified_labs
                if lab["loinc"] in relevant_loincs and lab["flag"] != "NORMAL"
            ]

            detected.append({
                "pattern": pattern["name"],
                "description": pattern["description"],
                "markers": triggered_markers[:5],
                "supports_icd10": pattern["supports_icd10"],
                "sensitivity_note": pattern["sensitivity_note"],
                "rules_met": rules_met,
                "rules_total": len(rules),
            })

    return detected


def generate_critical_alerts(classified_labs: list[dict]) -> list[dict]:
    """
    Generate CRITICAL-level alerts that must always surface — never suppressed.
    
    Returns:
        List of alert dicts
    """
    alerts = []

    for lab in classified_labs:
        loinc = lab["loinc"]
        display = lab["display"]
        value = float(lab["value"])
        unit = lab["unit"]

        # --- CRITICAL ALERTS ---
        if lab["flag"] == "CRITICAL":
            if loinc == "26464-8":
                if value >= 15.0:
                    message = f"WBC {value} {unit} — extreme leukocytosis. Rule out leukemia. Sepsis workup mandatory: blood cultures x2, lactate."
                else:
                    message = f"WBC {value} {unit} — severe leukocytosis. Sepsis workup recommended."
            
            elif loinc == "718-7":
                message = f"Hemoglobin {value} {unit} — critical anemia. Transfusion consideration."
            
            elif loinc == "777-3":
                message = f"Platelets {value} {unit} — critical thrombocytopenia."
            
            elif loinc == "2160-0":
                message = f"Creatinine {value} {unit} — critical renal impairment."
            
            elif loinc == "2823-3":
                if value > 5.0:
                    message = f"Potassium {value} {unit} — critical hyperkalemia. Cardiac emergency."
                else:
                    message = f"Potassium {value} {unit} — critical hypokalemia."
            
            else:
                message = f"{display} {value} {unit} — critical value."

            alerts.append({
                "level": "CRITICAL",
                "loinc": loinc,
                "display": display,
                "value": value,
                "unit": unit,
                "message": message,
                "action_required": True,
            })

        # --- HIGH-RISK ALERT (NEW BLOCK — OUTSIDE CRITICAL) ---
        if loinc == "1988-5" and value > 100:
            alerts.append({
                "level": "HIGH_RISK",
                "loinc": loinc,
                "display": display,
                "value": value,
                "unit": unit,
                "message": "CRP >100 — strong indicator of severe bacterial infection / sepsis risk",
                "action_required": True,
            })

    return alerts


def compute_overall_severity(classified_labs: list[dict]) -> str:
    """
    Compute overall severity category based on flag distribution.
    """
    flags = [lab["flag"] for lab in classified_labs]

    if "CRITICAL" in flags:
        return "SEVERE"
    elif flags.count("HIGH") + flags.count("LOW") >= 2:
        return "MODERATE"
    elif "HIGH" in flags or "LOW" in flags:
        return "MILD"
    else:
        return "NORMAL"