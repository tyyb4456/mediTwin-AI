# Agent 1: Patient Context Agent

**Role:** FHIR Data Layer — Entry Point of the System  
**Type:** A2A Agent + MCP Tool Consumer  
**Framework:** FastAPI + Python  
**Protocol:** SHARP context propagation → FHIR R4 REST API

---

## What This Agent Does

The Patient Context Agent is the **foundation of the entire MediTwin system**. Every other agent depends on the structured patient state this agent produces. It acts as the bridge between the Prompt Opinion platform (which provides SHARP context tokens) and the FHIR server (which holds raw patient data).

When the orchestrator triggers this agent, it receives a patient ID and a SHARP-issued FHIR access token. It then fetches all relevant FHIR resources, normalizes them into a unified patient state dictionary, and passes that object downstream to every specialist agent.

---

## Responsibilities

1. Extract patient ID and FHIR bearer token from SHARP context headers
2. Fetch the following FHIR R4 resources from the FHIR server:
   - `Patient` — demographics, identifiers
   - `Condition` — active and historical diagnoses (ICD-10 codes)
   - `MedicationRequest` — current and past prescriptions
   - `AllergyIntolerance` — drug and food allergies
   - `Observation` — lab results, vitals (LOINC codes)
   - `DiagnosticReport` — imaging reports, pathology
   - `Encounter` — recent hospital visits
3. Normalize all resources into a single structured `PatientState` Pydantic model
4. Cache the state in Redis (TTL: 10 minutes) to avoid redundant FHIR calls across agents
5. Return the `PatientState` object to the orchestrator

---

## Input

```json
{
  "patient_id": "fhir-patient-uuid",
  "fhir_base_url": "https://hapi.fhir.org/baseR4",
  "sharp_token": "Bearer eyJhbGciOiJSUzI1NiJ9..."
}
```

---

## Output

```json
{
  "patient_id": "fhir-patient-uuid",
  "demographics": {
    "name": "John Doe",
    "age": 54,
    "gender": "male",
    "dob": "1970-03-14"
  },
  "active_conditions": [
    { "code": "J18.9", "display": "Pneumonia, unspecified", "onset": "2025-04-01" }
  ],
  "medications": [
    { "drug": "Amoxicillin", "dose": "500mg", "frequency": "TID", "status": "active" }
  ],
  "allergies": [
    { "substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe" }
  ],
  "lab_results": [
    { "loinc": "26464-8", "display": "WBC", "value": 14.2, "unit": "10*3/uL", "reference_high": 11.0, "flag": "HIGH" }
  ],
  "diagnostic_reports": [...],
  "recent_encounters": [...],
  "state_timestamp": "2025-04-01T10:30:00Z"
}
```

---

## How It Works — Step by Step

### Step 1: Receive SHARP Context
The Prompt Opinion platform injects SHARP headers into every agent call. This agent reads:
- `X-SHARP-Patient-ID`
- `X-SHARP-FHIR-Token`
- `X-SHARP-FHIR-BaseURL`

### Step 2: Check Redis Cache
```python
cached = redis_client.get(f"patient_state:{patient_id}")
if cached:
    return PatientState.parse_raw(cached)
```

### Step 3: Parallel FHIR Fetch
Use `asyncio.gather()` to fetch all resource types concurrently — don't do sequential requests, it's too slow.
```python
results = await asyncio.gather(
    fetch_fhir(f"Patient/{patient_id}", token),
    fetch_fhir(f"Condition?patient={patient_id}&clinical-status=active", token),
    fetch_fhir(f"MedicationRequest?patient={patient_id}&status=active", token),
    fetch_fhir(f"AllergyIntolerance?patient={patient_id}", token),
    fetch_fhir(f"Observation?patient={patient_id}&category=laboratory&_sort=-date&_count=20", token),
    fetch_fhir(f"DiagnosticReport?patient={patient_id}&_sort=-date&_count=5", token),
)
```

### Step 4: Normalize
Map raw FHIR JSON into clean Pydantic models. Handle missing fields gracefully — FHIR data is inconsistent in real deployments.

### Step 5: Cache and Return
```python
redis_client.setex(f"patient_state:{patient_id}", 600, state.json())
return state
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.11 |
| API framework | FastAPI |
| FHIR client | `fhirclient` or raw `httpx` async |
| Data validation | Pydantic v2 |
| Caching | Redis |
| FHIR server (dev) | HAPI FHIR public sandbox |
| Auth | SHARP token passthrough |

---

## FHIR Server for Development

Use the public HAPI FHIR R4 sandbox — no account needed:
```
https://hapi.fhir.org/baseR4
```
Load synthetic patient data using Synthea (open-source synthetic patient generator).

---

## Your Existing Skills That Apply

- FastAPI REST API design (from your sales management app)
- Pydantic data modeling
- Redis caching (from AINutritionChef)
- Async Python with `asyncio`

---

## Common Pitfalls

- FHIR Observation bundles return pages — always follow `link.next` for complete lab history
- `MedicationRequest.status` can be `active`, `completed`, `stopped` — filter correctly
- AllergyIntolerance may have `clinicalStatus` of `active` or `inactive` — only send active ones downstream
- Always handle 404 gracefully — not every patient has every resource type