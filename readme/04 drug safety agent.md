# Agent 4: Drug Safety Agent (MCP Superpower)

**Role:** Medication Safety — Interaction Checking + Contraindication Detection  
**Type:** MCP Server (Superpower submission) + consumed as A2A tool  
**Framework:** FastAPI + MCP SDK  
**External APIs:** FDA OpenFDA Drug API (free, no auth required)

---

## What This Agent Does

The Drug Safety Agent is the system's **standalone MCP Superpower** — published to the Prompt Opinion Marketplace where any agent on the platform can call it, not just MediTwin. This is your dual-submission advantage: it serves as both an independent MCP tool (Superpower track) and a component of your full multi-agent system (Full Agent track).

It exposes three MCP tools that check medication safety:
1. `check_drug_interactions` — finds dangerous drug-drug interactions
2. `get_contraindications` — checks if a drug is contraindicated for a patient's conditions
3. `suggest_alternatives` — recommends safer alternatives when a conflict is detected

In the MediTwin workflow, this agent receives the Treatment Planning Agent's proposed medications and validates them against the patient's current medications, allergies, and conditions before any prescription recommendation is finalized.

---

## MCP Tools Exposed

### Tool 1: `check_drug_interactions`
```json
{
  "name": "check_drug_interactions",
  "description": "Check for dangerous interactions between a list of medications",
  "inputSchema": {
    "type": "object",
    "properties": {
      "medications": {
        "type": "array",
        "items": { "type": "string" },
        "description": "List of drug names or RxNorm codes",
        "example": ["Amoxicillin", "Warfarin", "Metformin"]
      },
      "patient_id": {
        "type": "string",
        "description": "Optional FHIR patient ID for SHARP context"
      }
    },
    "required": ["medications"]
  }
}
```

### Tool 2: `get_contraindications`
```json
{
  "name": "get_contraindications",
  "description": "Check if a medication is contraindicated for a patient's conditions or allergies",
  "inputSchema": {
    "type": "object",
    "properties": {
      "drug_name": { "type": "string", "description": "Drug name or RxNorm code" },
      "conditions": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Patient's active ICD-10 condition codes"
      },
      "allergies": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Patient's known allergens"
      }
    },
    "required": ["drug_name"]
  }
}
```

### Tool 3: `suggest_alternatives`
```json
{
  "name": "suggest_alternatives",
  "description": "Suggest safer medication alternatives when a drug has interactions or contraindications",
  "inputSchema": {
    "type": "object",
    "properties": {
      "drug_name": { "type": "string" },
      "reason_for_avoidance": { "type": "string" },
      "indication": { "type": "string", "description": "What condition the drug treats" },
      "patient_conditions": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["drug_name", "indication"]
  }
}
```

---

## Input (in MediTwin workflow)

```json
{
  "proposed_medications": ["Amoxicillin 500mg", "Ibuprofen 400mg"],
  "current_medications": ["Warfarin 5mg", "Metformin 850mg"],
  "patient_allergies": ["Penicillin"],
  "active_conditions": ["J18.9", "E11.9", "I48.0"]
}
```

---

## Output

```json
{
  "safety_status": "UNSAFE",
  "critical_interactions": [
    {
      "drug_a": "Amoxicillin",
      "drug_b": "Warfarin",
      "severity": "MODERATE",
      "mechanism": "Amoxicillin may increase anticoagulant effect of Warfarin by reducing gut flora that produces Vitamin K",
      "clinical_recommendation": "Monitor INR closely if co-administration is necessary. Consider dose adjustment.",
      "source": "FDA OpenFDA"
    }
  ],
  "contraindications": [
    {
      "drug": "Amoxicillin",
      "reason": "Patient allergy to Penicillin (cross-reactivity risk ~10%)",
      "severity": "HIGH",
      "recommendation": "AVOID — use macrolide antibiotic instead (e.g., Azithromycin)"
    }
  ],
  "alternatives": [
    {
      "replaces": "Amoxicillin",
      "alternative_drug": "Azithromycin 500mg",
      "rationale": "Macrolide antibiotic effective for community-acquired pneumonia; no penicillin cross-reactivity",
      "interaction_check_required": ["Warfarin", "Metformin"],
      "safe_to_prescribe": true
    }
  ],
  "approved_medications": ["Metformin 850mg"],
  "fhir_medication_requests": [...]
}
```

---

## How It Works — Step by Step

### Step 1: Resolve Drug Names to RxNorm
Use FDA OpenFDA to normalize drug names to RxNorm codes:
```python
async def resolve_rxnorm(drug_name: str) -> str:
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
    response = await httpx.get(url)
    return response.json()["results"][0]["openfda"]["rxcui"][0]
```

### Step 2: Check Interactions via FDA API
```python
async def check_interactions(rxnorm_codes: list[str]) -> dict:
    # Use RxNav interaction API (free, no auth)
    codes_str = "+".join(rxnorm_codes)
    url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={codes_str}"
    response = await httpx.get(url)
    return parse_interaction_pairs(response.json())
```

### Step 3: Check Allergy Cross-Reactivity
Build a local cross-reactivity lookup table:
```python
CROSS_REACTIVITY = {
    "Penicillin": ["Amoxicillin", "Ampicillin", "Nafcillin", "Oxacillin", "Piperacillin"],
    "Sulfa": ["Sulfamethoxazole", "Sulfadiazine", "Furosemide", "Thiazides"],
    "NSAIDs": ["Ibuprofen", "Naproxen", "Ketorolac", "Aspirin"]
}
```

### Step 4: Check Condition Contraindications
```python
CONTRAINDICATIONS = {
    "Ibuprofen": {
        "conditions": ["N18.9", "K25.0", "I50.9"],  # CKD, peptic ulcer, heart failure
        "reason": "NSAIDs worsen renal function, cause GI bleeding, exacerbate heart failure"
    },
    "Metformin": {
        "conditions": ["N18.4", "N18.5"],  # Stage 4-5 CKD
        "reason": "Risk of lactic acidosis in severe renal impairment"
    }
}
```

### Step 5: Generate Safe Alternatives
Use LLM only for this step — alternative drug suggestion requires clinical reasoning:
```python
prompt = f"""
Drug {drug_name} is contraindicated due to: {reason}
Patient's active conditions: {conditions}
Patient's other medications: {other_meds}

Suggest the safest 2-3 alternatives. Return JSON only.
"""
```

### Step 6: Return + Write FHIR MedicationRequests
Approved medications are returned as FHIR `MedicationRequest` resources.

---

## External APIs Used

| API | Purpose | Auth Required |
|---|---|---|
| FDA OpenFDA Drug Labels | Drug names, indications, warnings | No |
| NLM RxNav Interaction API | Drug-drug interaction pairs | No |
| NLM RxNorm API | Drug name normalization to codes | No |

All three are completely free with no API keys required in development.

---

## MCP Server Registration

Register on Prompt Opinion Marketplace with SHARP extension spec:
```json
{
  "name": "drug-safety-mcp",
  "version": "1.0.0",
  "description": "Drug interaction checker and contraindication detector using FDA data and FHIR patient context",
  "tools": ["check_drug_interactions", "get_contraindications", "suggest_alternatives"],
  "sharp_context": {
    "consumes": ["patient_id", "fhir_token"],
    "optional": true
  }
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| MCP server framework | FastAPI + MCP SDK |
| Drug interaction data | FDA OpenFDA + NLM RxNav |
| Cross-reactivity rules | Local Python dict (hardcoded, reviewed) |
| Condition contraindications | Local JSON database |
| Alternative suggestions | LangChain + LLM |
| Response caching | Redis (TTL: 1 hour — drug data changes rarely) |

---

## Your Existing Skills That Apply

- FastAPI REST API design — this is your strongest backend skill and the core of this agent
- Redis caching from AINutritionChef
- API integration from multiple hackathon projects
- Pydantic for request/response validation

---

## Why This Is Your Dual Submission

Publishing this as a standalone MCP Superpower means:
1. It appears independently in the Prompt Opinion Marketplace
2. Any other team's agent can discover and call it during the judging period
3. Judges see it both as a standalone tool AND as part of your system
4. It's the most practical, immediately deployable piece of your submission — real clinicians would use this today