# Agent 6: Digital Twin Agent

**Role:** Predictive Simulation — Future Outcome Modeling  
**Type:** A2A Agent  
**Framework:** Scikit-learn / XGBoost + LangChain + FastAPI  
**Data:** MIMIC-III Demo Dataset (public, no PHI)

---

## What This Agent Does

The Digital Twin Agent is **the concept that elevates MediTwin from a diagnostic tool to a decision support system**. It doesn't just ask "what is wrong?" — it simulates "what happens next?" under different treatment scenarios.

Given the full patient state plus the diagnosis and treatment recommendations from other agents, it builds a probabilistic model of the patient's likely outcomes:

- **Without treatment** — baseline disease progression risk
- **With Treatment Option A** — predicted response and complications
- **With Treatment Option B** — alternative scenario comparison

This "what-if" simulation is what judges will remember. No chatbot does this.

---

## Responsibilities

1. Receive synthesized patient state + proposed treatment options from the Orchestrator
2. Extract and engineer ML features from FHIR data
3. Run XGBoost risk models for: 30-day readmission, mortality risk, complication likelihood
4. Simulate each proposed treatment scenario
5. Return structured scenario comparison with probability estimates
6. Generate clinical narrative explaining the simulation results

---

## Input

```json
{
  "patient_state": { "...full PatientState..." },
  "diagnosis": "Community-acquired pneumonia (J18.9)",
  "treatment_options": [
    {
      "option_id": "A",
      "label": "Azithromycin 500mg + Supportive care",
      "drugs": ["Azithromycin 500mg OD x5 days"],
      "interventions": ["IV fluids", "O2 supplementation"]
    },
    {
      "option_id": "B",
      "label": "Ceftriaxone IV + Azithromycin (hospitalization)",
      "drugs": ["Ceftriaxone 1g IV OD", "Azithromycin 500mg OD"],
      "interventions": ["Hospitalization", "IV fluids", "Continuous monitoring"]
    },
    {
      "option_id": "C",
      "label": "No treatment (baseline)",
      "drugs": [],
      "interventions": []
    }
  ]
}
```

---

## Output

```json
{
  "simulation_summary": {
    "patient_risk_profile": "MODERATE-HIGH",
    "primary_concern": "Age 54 + elevated WBC suggests moderate-severity pneumonia with sepsis risk",
    "recommended_option": "B",
    "recommendation_confidence": 0.81
  },
  "scenarios": [
    {
      "option_id": "A",
      "label": "Azithromycin outpatient",
      "predictions": {
        "recovery_probability_7d": 0.71,
        "readmission_risk_30d": 0.24,
        "complication_risk": 0.31,
        "mortality_risk_30d": 0.04,
        "estimated_recovery_days": 8
      },
      "key_risks": ["Outpatient management risk — monitor for deterioration", "Sepsis not fully covered"],
      "suitable_for_outpatient": false
    },
    {
      "option_id": "B",
      "label": "IV antibiotics + hospitalization",
      "predictions": {
        "recovery_probability_7d": 0.89,
        "readmission_risk_30d": 0.09,
        "complication_risk": 0.12,
        "mortality_risk_30d": 0.02,
        "estimated_recovery_days": 5
      },
      "key_risks": ["Hospital-acquired infection risk", "Cost implications"],
      "suitable_for_outpatient": false
    },
    {
      "option_id": "C",
      "label": "No treatment (baseline)",
      "predictions": {
        "recovery_probability_7d": 0.21,
        "readmission_risk_30d": 0.72,
        "complication_risk": 0.81,
        "mortality_risk_30d": 0.18,
        "estimated_recovery_days": null
      },
      "key_risks": ["High mortality risk", "Near-certain disease progression", "Sepsis likely"],
      "suitable_for_outpatient": false
    }
  ],
  "what_if_narrative": "For this 54-year-old male presenting with moderate pneumonia, Option B (IV antibiotic hospitalization) provides an 89% 7-day recovery probability versus 71% for outpatient oral therapy. The elevated WBC (18.4) and age increase sepsis risk, making inpatient management the safer choice. Without treatment, 30-day mortality risk reaches 18%.",
  "fhir_care_plan": {
    "resourceType": "CarePlan",
    "status": "active",
    "intent": "plan",
    "title": "AI-Recommended Treatment Plan (Option B)",
    "...": "..."
  }
}
```

---

## How It Works — Step by Step

### Step 1: Feature Engineering from FHIR Data
Extract ML features from the normalized `PatientState`:
```python
def engineer_features(patient_state: PatientState, diagnosis: str) -> np.ndarray:
    return {
        # Demographics
        "age": patient_state.demographics.age,
        "gender_male": 1 if patient_state.demographics.gender == "male" else 0,
        
        # Vital signs from Observations
        "wbc": get_lab_value(patient_state.labs, "26464-8"),  # WBC
        "creatinine": get_lab_value(patient_state.labs, "2160-0"),
        "albumin": get_lab_value(patient_state.labs, "1751-7"),
        "glucose": get_lab_value(patient_state.labs, "2345-7"),
        
        # Comorbidity burden (count of active ICD-10 conditions)
        "comorbidity_count": len(patient_state.active_conditions),
        "has_diabetes": has_condition(patient_state, ["E11", "E10"]),
        "has_ckd": has_condition(patient_state, ["N18"]),
        "has_chf": has_condition(patient_state, ["I50"]),
        "has_copd": has_condition(patient_state, ["J44"]),
        
        # Current medications (polypharmacy risk)
        "med_count": len(patient_state.medications),
        
        # Clinical severity proxy
        "critical_lab_count": count_critical_labs(patient_state.labs),
    }
```

### Step 2: Risk Model Inference
Use pre-trained XGBoost models (one per outcome):
```python
models = {
    "readmission_30d": xgb.XGBClassifier(),   # loaded from file
    "mortality_30d": xgb.XGBClassifier(),      # loaded from file
    "complication": xgb.XGBClassifier()         # loaded from file
}

def predict_baseline_risks(features: dict) -> dict:
    X = pd.DataFrame([features])
    return {
        "readmission_risk_30d": float(models["readmission_30d"].predict_proba(X)[0][1]),
        "mortality_risk_30d": float(models["mortality_30d"].predict_proba(X)[0][1]),
        "complication_risk": float(models["complication"].predict_proba(X)[0][1])
    }
```

### Step 3: Treatment Effect Simulation
Apply evidence-based treatment effect modifiers on top of baseline risks:
```python
TREATMENT_EFFECTS = {
    "Azithromycin": {
        "readmission_30d_reduction": 0.35,
        "mortality_30d_reduction": 0.55,
        "complication_reduction": 0.40,
        "recovery_days_reduction": 3
    },
    "Ceftriaxone_IV": {
        "readmission_30d_reduction": 0.62,
        "mortality_30d_reduction": 0.72,
        "complication_reduction": 0.61,
        "recovery_days_reduction": 5
    }
}

def simulate_treatment(baseline_risks: dict, treatment: TreatmentOption) -> dict:
    simulated = baseline_risks.copy()
    
    for drug in treatment.drugs:
        drug_name = parse_drug_name(drug)
        effect = TREATMENT_EFFECTS.get(drug_name, {})
        
        simulated["readmission_risk_30d"] *= (1 - effect.get("readmission_30d_reduction", 0))
        simulated["mortality_risk_30d"] *= (1 - effect.get("mortality_30d_reduction", 0))
        simulated["complication_risk"] *= (1 - effect.get("complication_reduction", 0))
    
    simulated["recovery_probability_7d"] = 1 - simulated["mortality_risk_30d"] - simulated["complication_risk"] * 0.5
    
    return simulated
```

### Step 4: LLM Narrative Generation
```python
prompt = f"""
You are a clinical decision support system explaining a treatment simulation to a physician.

Patient: {age}y {gender}, Diagnosis: {diagnosis}
Risk profile: {risk_profile}

Scenario comparison:
{scenario_json}

Write a concise 3-sentence clinical narrative comparing the scenarios and recommending the best option.
Focus on numbers and clinical reasoning. No filler phrases.
"""
```

### Step 5: Output FHIR CarePlan
Package the recommended scenario as a FHIR `CarePlan` resource:
```python
fhir_care_plan = {
    "resourceType": "CarePlan",
    "status": "active",
    "intent": "plan",
    "subject": {"reference": f"Patient/{patient_id}"},
    "title": f"AI-Recommended Treatment Plan — {recommended_option.label}",
    "description": narrative,
    "activity": [
        {
            "detail": {
                "kind": "MedicationRequest",
                "code": {"text": drug},
                "status": "scheduled"
            }
        }
        for drug in recommended_option.drugs
    ]
}
```

---

## Training the Risk Models

### Dataset: MIMIC-III Demo (Free, No PHI)
MIMIC-III Demo is a publicly available subset of the MIMIC-III clinical database — 100 de-identified patients, completely free with no credentialing required.

Download: `https://physionet.org/content/mimiciii-demo/1.4/`

### Training Pipeline
```python
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load MIMIC-III demo tables
admissions = pd.read_csv("ADMISSIONS.csv")
patients = pd.read_csv("PATIENTS.csv")
diagnoses = pd.read_csv("DIAGNOSES_ICD.csv")
labevents = pd.read_csv("LABEVENTS.csv")

# Engineer features (age, comorbidities, labs)
features_df = engineer_features_from_mimic(admissions, patients, diagnoses, labevents)

# Target: 30-day readmission
features_df["readmitted_30d"] = compute_readmission_label(admissions)

# Train
X = features_df.drop("readmitted_30d", axis=1)
y = features_df["readmitted_30d"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)
model.save_model("models/readmission_30d.json")
```

Note: The MIMIC-III demo only has 100 patients — the model won't be production-accurate, but it will produce plausible predictions for demo purposes. Be transparent about this in your submission.

---

## Tech Stack

| Component | Technology |
|---|---|
| Risk models | XGBoost (pre-trained, loaded at startup) |
| Feature engineering | Pandas + NumPy |
| Treatment simulation | Rules-based Python (treatment effect modifiers) |
| Narrative generation | LangChain + LLM |
| FHIR output | FHIR R4 CarePlan builder |
| API framework | FastAPI |
| Data | MIMIC-III Demo Dataset |

---

## Your Existing Skills That Apply

- XGBoost and LightGBM from your classification projects
- Feature engineering and cross-validation from ML pipeline work
- SHAP explainability from bank churn analysis — apply same technique here to explain why the model scored a patient as high-risk
- Scikit-learn pipelines

---

## Honest Framing for Judges

When presenting this agent, be transparent:
- Models are trained on MIMIC-III demo (100 patients) — representative, not production-scale
- Treatment effect modifiers are literature-based heuristics, not RCT-derived
- This demonstrates the *architecture* of a digital twin — real deployment would require population-scale training data

Judges respect intellectual honesty. Overclaiming model accuracy would be a red flag; demonstrating the right architecture with correct framing wins.