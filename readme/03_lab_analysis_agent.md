# Agent 3: Lab Analysis Agent

**Role:** Pathology Interpretation — Structured Abnormality Detection  
**Type:** A2A Agent  
**Framework:** FastAPI + Python (rules engine) + LLM interpretation layer  
**Protocol:** Receives PatientState from Orchestrator; outputs to Consensus Agent

---

## What This Agent Does

The Lab Analysis Agent reads a patient's FHIR `Observation` resources (lab values, vitals) and does two things:

1. **Flags abnormalities** — systematically compares each result against clinical reference ranges and marks values as HIGH, LOW, CRITICAL, or NORMAL
2. **Interprets combinations** — uses LLM reasoning to identify patterns that emerge from multiple abnormal values together (e.g., "high WBC + high CRP + fever = systemic bacterial infection")

Its output is designed to either confirm or challenge the Diagnosis Agent's conclusions, feeding directly into the disagreement detection system.

---

## Responsibilities

1. Parse all `Observation` resources from `PatientState`
2. Classify each value against age/gender-adjusted reference ranges
3. Flag CRITICAL values (require immediate escalation regardless of other agents)
4. Identify multi-marker patterns and generate clinical interpretation
5. Output structured lab report as annotated FHIR `Observation` resources
6. Send comparison signal to Consensus Agent

---

## Input

```json
{
  "patient_state": { "...full PatientState..." },
  "diagnosis_agent_output": {
    "top_diagnosis": "Community-acquired pneumonia",
    "confidence": 0.87
  }
}
```

---

## Output

```json
{
  "lab_summary": {
    "total_results": 12,
    "abnormal_count": 4,
    "critical_count": 1,
    "overall_severity": "MODERATE"
  },
  "flagged_results": [
    {
      "loinc": "26464-8",
      "display": "White Blood Cell Count",
      "value": 18.4,
      "unit": "10*3/uL",
      "reference_range": "4.5 - 11.0",
      "flag": "CRITICAL",
      "clinical_significance": "Severely elevated WBC strongly indicates active bacterial infection or significant inflammatory response"
    },
    {
      "loinc": "2157-6",
      "display": "Creatine Kinase",
      "value": 420,
      "unit": "U/L",
      "reference_range": "30 - 200",
      "flag": "HIGH",
      "clinical_significance": "Mildly elevated CK — monitor for muscle involvement"
    }
  ],
  "pattern_analysis": {
    "identified_patterns": [
      {
        "pattern": "Bacterial infection markers",
        "markers": ["WBC CRITICAL", "CRP HIGH", "Neutrophil % HIGH"],
        "interpretation": "Multi-marker pattern strongly consistent with active bacterial infection. Combined sensitivity for bacterial pneumonia: ~89%.",
        "supports_diagnosis": "J18.9 - Pneumonia"
      }
    ]
  },
  "diagnosis_confirmation": {
    "confirms_top_diagnosis": true,
    "diagnosis_code": "J18.9",
    "lab_confidence_boost": 0.12,
    "reasoning": "Lab pattern (high WBC, elevated CRP, neutrophilia) strongly supports bacterial pneumonia diagnosis"
  },
  "critical_alerts": [
    {
      "level": "CRITICAL",
      "message": "WBC 18.4 — Sepsis workup recommended. Order blood cultures immediately.",
      "action_required": true
    }
  ]
}
```

---

## How It Works — Step by Step

### Step 1: Parse and Classify Observations
For each FHIR Observation:
```python
def classify_result(obs: FHIRObservation, patient_age: int, patient_gender: str) -> LabResult:
    ref_range = get_reference_range(obs.loinc_code, patient_age, patient_gender)
    value = obs.value_quantity
    
    if value > ref_range.critical_high:
        flag = "CRITICAL"
    elif value > ref_range.high:
        flag = "HIGH"
    elif value < ref_range.critical_low:
        flag = "CRITICAL"
    elif value < ref_range.low:
        flag = "LOW"
    else:
        flag = "NORMAL"
    
    return LabResult(loinc=obs.loinc_code, value=value, flag=flag, ...)
```

### Step 2: Reference Range Lookup
Use a local reference range database (JSON file or SQLite):
```python
# reference_ranges.json structure
{
  "26464-8": {  # WBC
    "display": "White Blood Cell Count",
    "unit": "10*3/uL",
    "adult_male": {"low": 4.5, "high": 11.0, "critical_high": 30.0},
    "adult_female": {"low": 4.5, "high": 11.0, "critical_high": 30.0},
    "pediatric": {"low": 5.0, "high": 15.0, "critical_high": 30.0}
  }
}
```

Key LOINC codes to support (minimum viable set):
| LOINC | Test | Clinical Significance |
|---|---|---|
| 26464-8 | WBC | Infection, immune status |
| 718-7 | Hemoglobin | Anemia |
| 777-3 | Platelets | Coagulation, bleeding risk |
| 2160-0 | Creatinine | Kidney function |
| 1742-6 | ALT | Liver function |
| 2345-7 | Glucose | Diabetes, metabolic |
| 1751-7 | Albumin | Nutritional status, liver |
| 2823-3 | Potassium | Electrolyte balance |
| 6598-7 | Troponin T | Cardiac injury |
| 1988-5 | CRP | Inflammation, infection |

### Step 3: Pattern Detection (Rules Engine)
Define clinical patterns as rule sets:
```python
PATTERNS = [
    {
        "name": "Bacterial infection markers",
        "rules": [
            lambda labs: labs["26464-8"].flag in ["HIGH", "CRITICAL"],  # WBC high
            lambda labs: labs["1988-5"].flag in ["HIGH", "CRITICAL"],   # CRP high
        ],
        "min_rules_met": 2,
        "interpretation": "Multi-marker pattern consistent with bacterial infection",
        "supports_icd10": ["J18.9", "A41.9"]
    },
    {
        "name": "Acute kidney injury",
        "rules": [
            lambda labs: labs["2160-0"].flag in ["HIGH", "CRITICAL"],   # Creatinine
            lambda labs: labs["2823-3"].flag in ["HIGH", "CRITICAL"],   # Potassium
        ],
        "min_rules_met": 1,
        "interpretation": "Possible acute kidney injury — urgent nephrology review",
        "supports_icd10": ["N17.9"]
    }
]
```

### Step 4: LLM Interpretation Layer
For complex patterns not covered by rules, use LLM:
```python
prompt = f"""
You are a clinical pathologist interpreting lab results.

ABNORMAL RESULTS:
{abnormal_results_json}

DIAGNOSIS AGENT CONCLUSION:
{diagnosis_agent_output}

Do the lab findings support, contradict, or add nuance to this diagnosis?
Return structured JSON with confirmation status and clinical reasoning.
"""
```

### Step 5: Critical Alert Generation
Any CRITICAL flag triggers an immediate alert regardless of other agent outputs — the Orchestrator cannot suppress a CRITICAL lab alert.

---

## Reference Range Database

Build a `reference_ranges.json` from these free sources:
- ARUP Laboratories reference ranges (publicly available)
- Mayo Clinic Lab Test Reference Intervals
- LOINC database (loinc.org — free download)

---

## Tech Stack

| Component | Technology |
|---|---|
| Core logic | Python rules engine (pure Python) |
| Reference ranges | Local JSON database (SQLite optional) |
| LLM interpretation | LangChain + GPT-4o-mini |
| FHIR parsing | `fhirclient` Python library |
| API framework | FastAPI |

---

## Your Existing Skills That Apply

- XGBoost + SHAP for feature importance — the pattern detection is the same logic as your feature engineering work
- Heart disease prediction project — lab-based risk classification is exactly this
- Scikit-learn pipelines — the classification logic maps directly here

---

## Critical Design Decision

Keep the rules engine and the LLM layer separate. The rules engine runs first and is deterministic — it handles CRITICAL flags. The LLM only runs after to add qualitative interpretation. This ensures CRITICAL alerts are never suppressed by LLM hallucination.