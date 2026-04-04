"""
Agent 1: Patient Context Agent
Entry point of MediTwin - fetches and normalizes FHIR data
"""
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# Add parent directory to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.models import PatientState, Demographics, Condition, Medication, Allergy, LabResult, DiagnosticReport, Encounter
from shared.redis_client import redis_client
from shared.sharp_context import SHARPContext


# Request/Response models
class PatientContextRequest(BaseModel):
    """Request to fetch patient context"""
    patient_id: str
    fhir_base_url: Optional[str] = "https://hapi.fhir.org/baseR4"
    sharp_token: Optional[str] = None


class PatientContextResponse(BaseModel):
    """Response with patient state"""
    patient_state: PatientState
    cache_hit: bool
    fetch_time_ms: int


# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - setup and teardown"""
    global http_client
    
    # Startup
    http_client = httpx.AsyncClient(timeout=30.0)
    await redis_client.connect()
    print("✓ Patient Context Agent started - HTTP client and Redis connected")
    
    yield
    
    # Shutdown
    await http_client.aclose()
    await redis_client.disconnect()
    print("✓ Patient Context Agent shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MediTwin Patient Context Agent",
    description="FHIR data ingestion and normalization layer",
    version="1.0.0",
    lifespan=lifespan
)


async def fetch_fhir_resource(
    resource_type: str,
    resource_id: Optional[str] = None,
    search_params: Optional[dict] = None,
    headers: Optional[dict] = None
) -> dict:
    """
    Fetch FHIR resource from server
    
    Args:
        resource_type: FHIR resource type (e.g., 'Patient', 'Condition')
        resource_id: Specific resource ID (optional)
        search_params: Query parameters for search (optional)
        headers: Auth headers (optional)
    """
    fhir_base_url = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
    
    if resource_id:
        url = f"{fhir_base_url}/{resource_type}/{resource_id}"
    else:
        url = f"{fhir_base_url}/{resource_type}"
    
    try:
        response = await http_client.get(
            url,
            params=search_params or {},
            headers=headers or {}
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"FHIR fetch error for {resource_type}: {e}")
        return {}


def normalize_patient(fhir_patient: dict) -> Demographics:
    """Extract demographics from FHIR Patient resource"""
    if not fhir_patient or fhir_patient.get("resourceType") != "Patient":
        raise ValueError("Invalid Patient resource")
    
    # Extract name
    name = "Unknown"
    if fhir_patient.get("name"):
        name_parts = fhir_patient["name"][0]
        given = " ".join(name_parts.get("given", []))
        family = name_parts.get("family", "")
        name = f"{given} {family}".strip()
    
    # Extract birth date and calculate age
    birth_date = fhir_patient.get("birthDate", "1970-01-01")
    birth_year = int(birth_date.split("-")[0])
    current_year = datetime.now().year
    age = current_year - birth_year
    
    # Extract gender
    gender = fhir_patient.get("gender", "unknown")
    
    return Demographics(
        name=name,
        age=age,
        gender=gender,
        dob=birth_date
    )


def normalize_conditions(fhir_bundle: dict) -> list[Condition]:
    """Extract active conditions from FHIR Bundle"""
    conditions = []
    
    if not fhir_bundle.get("entry"):
        return conditions
    
    for entry in fhir_bundle["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        
        # Only active conditions
        clinical_status = resource.get("clinicalStatus", {})
        status_code = clinical_status.get("coding", [{}])[0].get("code", "")
        if status_code != "active":
            continue
        
        # Extract ICD-10 code
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]
        
        conditions.append(Condition(
            code=coding.get("code", "UNKNOWN"),
            display=coding.get("display", "Unknown condition"),
            onset=resource.get("onsetDateTime", resource.get("onsetString"))
        ))
    
    return conditions


def normalize_medications(fhir_bundle: dict) -> list[Medication]:
    """Extract medications from FHIR Bundle"""
    medications = []
    
    if not fhir_bundle.get("entry"):
        return medications
    
    for entry in fhir_bundle["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "MedicationRequest":
            continue
        
        # Extract medication name
        med_concept = resource.get("medicationCodeableConcept", {})
        coding = med_concept.get("coding", [{}])[0]
        drug_name = coding.get("display", "Unknown medication")
        
        # Extract dosage
        dosage = resource.get("dosageInstruction", [{}])[0] if resource.get("dosageInstruction") else {}
        dose = dosage.get("text", "")
        
        # Extract status
        status = resource.get("status", "unknown")
        
        medications.append(Medication(
            drug=drug_name,
            dose=dose,
            frequency="",
            status=status
        ))
    
    return medications


def normalize_allergies(fhir_bundle: dict) -> list[Allergy]:
    """Extract allergies from FHIR Bundle"""
    allergies = []
    
    if not fhir_bundle.get("entry"):
        return allergies
    
    for entry in fhir_bundle["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "AllergyIntolerance":
            continue
        
        # Extract allergen
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]
        substance = coding.get("display", "Unknown allergen")
        
        # Extract reaction
        reactions = resource.get("reaction", [])
        reaction = reactions[0].get("manifestation", [{}])[0].get("text", "") if reactions else ""
        
        # Extract severity
        criticality = resource.get("criticality", "unknown")
        
        allergies.append(Allergy(
            substance=substance,
            reaction=reaction,
            severity=criticality
        ))
    
    return allergies


def normalize_observations(fhir_bundle: dict) -> list[LabResult]:
    """Extract lab results from FHIR Observation Bundle"""
    labs = []
    
    if not fhir_bundle.get("entry"):
        return labs
    
    for entry in fhir_bundle["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue
        
        # Extract LOINC code
        code_obj = resource.get("code", {})
        coding = code_obj.get("coding", [{}])[0]
        loinc = coding.get("code", "UNKNOWN")
        display = coding.get("display", "Unknown lab")
        
        # Extract value
        value_quantity = resource.get("valueQuantity", {})
        value = value_quantity.get("value")
        unit = value_quantity.get("unit", "")
        
        if value is None:
            continue
        
        # Extract reference range
        ref_range = resource.get("referenceRange", [{}])[0] if resource.get("referenceRange") else {}
        ref_low = ref_range.get("low", {}).get("value")
        ref_high = ref_range.get("high", {}).get("value")
        
        # Determine flag
        flag = "NORMAL"
        if ref_high and value > ref_high:
            flag = "HIGH"
        elif ref_low and value < ref_low:
            flag = "LOW"
        
        labs.append(LabResult(
            loinc=loinc,
            display=display,
            value=float(value),
            unit=unit,
            reference_high=ref_high,
            reference_low=ref_low,
            flag=flag
        ))
    
    return labs


@app.post("/fetch", response_model=PatientContextResponse)
async def fetch_patient_context(
    request: PatientContextRequest,
    x_sharp_patient_id: Optional[str] = Header(None),
    x_sharp_fhir_token: Optional[str] = Header(None),
    x_sharp_fhir_base_url: Optional[str] = Header(None)
) -> PatientContextResponse:
    """
    Fetch and normalize patient FHIR data
    
    Priority: SHARP headers > request body
    """
    start_time = datetime.now()
    
    # Resolve patient ID and FHIR config
    patient_id = x_sharp_patient_id or request.patient_id
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url
    auth_headers = {"Authorization": x_sharp_fhir_token} if x_sharp_fhir_token else {}
    
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id required")
    
    # Check Redis cache
    cache_key = f"patient_state:{patient_id}"
    cached = await redis_client.get_json(cache_key)
    
    if cached:
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return PatientContextResponse(
            patient_state=PatientState(**cached),
            cache_hit=True,
            fetch_time_ms=int(elapsed)
        )
    
    # Parallel FHIR fetches
    import asyncio
    
    patient_task = fetch_fhir_resource("Patient", patient_id, headers=auth_headers)
    conditions_task = fetch_fhir_resource(
        "Condition",
        search_params={"patient": patient_id, "clinical-status": "active"},
        headers=auth_headers
    )
    medications_task = fetch_fhir_resource(
        "MedicationRequest",
        search_params={"patient": patient_id, "status": "active"},
        headers=auth_headers
    )
    allergies_task = fetch_fhir_resource(
        "AllergyIntolerance",
        search_params={"patient": patient_id},
        headers=auth_headers
    )
    observations_task = fetch_fhir_resource(
        "Observation",
        search_params={
            "patient": patient_id,
            "category": "laboratory",
            "_sort": "-date",
            "_count": "20"
        },
        headers=auth_headers
    )
    
    # Await all parallel fetches
    patient_data, conditions_bundle, medications_bundle, allergies_bundle, observations_bundle = \
        await asyncio.gather(
            patient_task,
            conditions_task,
            medications_task,
            allergies_task,
            observations_task
        )
    
    # Normalize data
    try:
        demographics = normalize_patient(patient_data)
        conditions = normalize_conditions(conditions_bundle)
        medications = normalize_medications(medications_bundle)
        allergies = normalize_allergies(allergies_bundle)
        labs = normalize_observations(observations_bundle)
        
        # Build PatientState
        patient_state = PatientState(
            patient_id=patient_id,
            demographics=demographics,
            active_conditions=conditions,
            medications=medications,
            allergies=allergies,
            lab_results=labs,
            diagnostic_reports=[],
            recent_encounters=[],
            state_timestamp=datetime.now().isoformat() + "Z",
            imaging_available=False
        )
        
        # Cache for 10 minutes
        await redis_client.set_json(cache_key, patient_state.model_dump(), ttl=600)
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        
        return PatientContextResponse(
            patient_state=patient_state,
            cache_hit=False,
            fetch_time_ms=int(elapsed)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FHIR normalization error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "patient-context",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)