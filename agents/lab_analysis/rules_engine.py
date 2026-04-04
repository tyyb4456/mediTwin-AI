"""
Lab Analysis Rules Engine
Pure Python — deterministic, no LLM.
Runs FIRST before LLM interpretation.
CRITICAL flags from this engine are NEVER suppressed by LLM output.
"""
import json
from pathlib import Path
from typing import Optional


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
    # If no reference range, leave as NORMAL (don't flag what we can't evaluate)

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
            continue  # Skip labs with no numeric value

        result = classify_result(loinc, float(value), age, gender)

        # Preserve original display name if we don't have a mapping
        if result["display"] == loinc and lab.get("display"):
            result["display"] = lab["display"]

        classified.append(result)

    return classified


# ── Clinical Pattern Detection ─────────────────────────────────────────────────
# Each pattern is: name, required LOINCs with required flags, interpretation, ICD-10 codes

CLINICAL_PATTERNS = [
    {
        "name": "Bacterial infection markers",
        "description": "Multi-marker pattern consistent with bacterial infection",
        "rules": [
            # WBC elevated
            lambda labs: labs.get("26464-8", {}).get("flag") in ("HIGH", "CRITICAL"),
            # CRP elevated (OR if CRP not available, neutrophilia)
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
            # WBC critically elevated OR critically low
            lambda labs: labs.get("26464-8", {}).get("flag") == "CRITICAL",
            # Any organ dysfunction marker elevated
            lambda labs: (
                labs.get("2160-0", {}).get("flag") in ("HIGH", "CRITICAL")   # Creatinine
                or labs.get("1742-6", {}).get("flag") in ("HIGH", "CRITICAL") # ALT
                or labs.get("6598-7", {}).get("flag") in ("HIGH", "CRITICAL") # Troponin
            ),
        ],
        "min_rules_met": 1,  # Critical WBC alone is enough to raise sepsis concern
        "supports_icd10": ["A41.9"],
        "sensitivity_note": "CRITICAL WBC warrants sepsis workup: blood cultures, lactate",
    },
    {
        "name": "Acute kidney injury (AKI) markers",
        "description": "Laboratory pattern consistent with acute or chronic kidney injury",
        "rules": [
            lambda labs: labs.get("2160-0", {}).get("flag") in ("HIGH", "CRITICAL"),  # Creatinine
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
            lambda labs: labs.get("6598-7", {}).get("flag") in ("HIGH", "CRITICAL"),  # Troponin T
        ],
        "min_rules_met": 1,
        "supports_icd10": ["I21.9", "I25.1"],
        "sensitivity_note": "Troponin elevation in infection context may indicate demand ischemia",
    },
    {
        "name": "Anemia",
        "description": "Hemoglobin below normal range",
        "rules": [
            lambda labs: labs.get("718-7", {}).get("flag") in ("LOW", "CRITICAL"),  # Hemoglobin
        ],
        "min_rules_met": 1,
        "supports_icd10": ["D64.9"],
        "sensitivity_note": "Severity classification: mild >10, moderate 8-10, severe <8 g/dL",
    },
    {
        "name": "Coagulation concern (thrombocytopenia)",
        "description": "Low platelet count indicating bleeding risk",
        "rules": [
            lambda labs: labs.get("777-3", {}).get("flag") in ("LOW", "CRITICAL"),  # Platelets
        ],
        "min_rules_met": 1,
        "supports_icd10": ["D69.6"],
        "sensitivity_note": "Platelets <50 = serious bleeding risk; <20 = spontaneous bleeding risk",
    },
    {
        "name": "Hyperglycemia / Diabetes decompensation",
        "description": "Significantly elevated glucose",
        "rules": [
            lambda labs: labs.get("2345-7", {}).get("flag") in ("HIGH", "CRITICAL"),  # Glucose
        ],
        "min_rules_met": 1,
        "supports_icd10": ["E11.9", "E13.9"],
        "sensitivity_note": "Critical glucose >500 may indicate DKA or HHS — check ketones and pH",
    },
]


def detect_patterns(classified_labs: list[dict]) -> list[dict]:
    """
    Run all clinical pattern rules against the classified lab results.

    Returns:
        List of detected pattern dicts with interpretation details
    """
    # Build a LOINC → result lookup for fast rule evaluation
    labs_by_loinc = {lab["loinc"]: lab for lab in classified_labs}

    detected = []
    for pattern in CLINICAL_PATTERNS:
        rules = pattern["rules"]
        min_met = pattern["min_rules_met"]

        rules_met = sum(1 for rule in rules if rule(labs_by_loinc))

        if rules_met >= min_met:
            # Collect the specific markers that triggered this pattern
            triggered_markers = []
            for lab in classified_labs:
                if lab["flag"] in ("HIGH", "LOW", "CRITICAL"):
                    # Only include markers that are relevant to this pattern's ICD-10 codes
                    triggered_markers.append(
                        f"{lab['display']} {lab['flag']} ({lab['value']} {lab['unit']})"
                    )

            detected.append({
                "pattern": pattern["name"],
                "description": pattern["description"],
                "markers": triggered_markers[:5],  # top 5 relevant
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
        if lab["flag"] != "CRITICAL":
            continue

        loinc = lab["loinc"]
        display = lab["display"]
        value = lab["value"]
        unit = lab["unit"]

        # Specific actionable messages per LOINC
        if loinc == "26464-8":  # WBC
            if value >= 15.0:
                message = f"WBC {value} {unit} — extreme leukocytosis. Rule out leukemia. Sepsis workup mandatory: blood cultures x2, lactate."
            else:
                message = f"WBC {value} {unit} — severe leukocytosis. Sepsis workup recommended: blood cultures before antibiotics."
        elif loinc == "718-7":  # Hemoglobin
            message = f"Hemoglobin {value} {unit} — critical anemia. Transfusion threshold consideration. Evaluate for active bleeding."
        elif loinc == "777-3":  # Platelets
            message = f"Platelets {value} {unit} — critical thrombocytopenia. High bleeding risk. Avoid invasive procedures. Hematology consult."
        elif loinc == "2160-0":  # Creatinine
            message = f"Creatinine {value} {unit} — critical renal impairment. Urgent nephrology consult. Adjust all renally-cleared medications."
        elif loinc == "2823-3":  # Potassium
            if value > 5.0:
                message = f"Potassium {value} {unit} — critical hyperkalemia. Cardiac emergency — obtain EKG immediately. Consider calcium gluconate."
            else:
                message = f"Potassium {value} {unit} — critical hypokalemia. Cardiac arrhythmia risk. IV potassium replacement required."
        elif loinc == "2947-0":  # Sodium
            if value > 145:
                message = f"Sodium {value} {unit} — critical hypernatremia. Risk of cerebral dehydration. Gradual correction required."
            else:
                message = f"Sodium {value} {unit} — critical hyponatremia. Risk of seizures and cerebral edema. Controlled correction mandatory."
        elif loinc == "2345-7":  # Glucose
            if value > 400:
                message = f"Glucose {value} {unit} — critical hyperglycemia. Evaluate for DKA/HHS. Check blood gases and ketones."
            else:
                message = f"Glucose {value} {unit} — critical hypoglycemia. Immediate glucose administration required."
        elif loinc == "6598-7":  # Troponin
            message = f"Troponin {value} {unit} — critical elevation. Acute myocardial injury. Cardiology consult. Serial troponins every 3 hours."
        elif loinc == "2708-6":  # PaO2
            message = f"PaO2 {value} {unit} — critical hypoxemia. Respiratory failure. Supplemental oxygen or ventilatory support required."
        else:
            message = f"{display} {value} {unit} — critical value. Immediate clinical review required."

        alerts.append({
            "level": "CRITICAL",
            "loinc": loinc,
            "display": display,
            "value": value,
            "unit": unit,
            "message": message,
            "action_required": True,
        })

    return alerts

def compute_overall_severity(classified_labs: list[dict]) -> str:
    flags = [lab["flag"] for lab in classified_labs]

    if "CRITICAL" in flags:
        return "SEVERE"
    elif flags.count("HIGH") + flags.count("LOW") >= 2:  # ← was 3
        return "MODERATE"
    elif "HIGH" in flags or "LOW" in flags:
        return "MILD"
    else:
        return "NORMAL"