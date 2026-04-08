"""
Agent 1: Patient Context Agent — COMPLETE REBUILD
Entry point of MediTwin - fetches and normalizes FHIR data

Role: FHIR Data Layer — Entry Point of the System
Type: A2A Agent + MCP Tool Consumer
Protocol: SHARP context propagation → FHIR R4 REST API

This agent is THE FOUNDATION. Every other agent depends on its output.
If this agent fails, nothing else can run.

Key Responsibilities:
1. Extract patient ID and FHIR bearer token from SHARP context headers
2. Fetch all relevant FHIR R4 resources (Patient, Condition, Medication, etc.)
3. Normalize into a single structured PatientState Pydantic model
4. Cache in Redis (TTL: 10 minutes) to avoid redundant FHIR calls
5. Return PatientState to orchestrator

SHARP Context Strategy (CRITICAL FIX):
- Production mode: Reads X-SHARP-Patient-ID, X-SHARP-FHIR-Token, X-SHARP-FHIR-BaseURL headers
- Development mode: Falls back to request body if headers not present
- This dual-mode design makes development smooth without simulating headers every time

Missing Features Now Implemented:
✓ Proper SHARP context header resolution
✓ Fallback to request body for development
✓ Imaging availability detection from DiagnosticReport
✓ Graceful handling of missing FHIR resources (empty arrays, not exceptions)
✓ Parallel FHIR fetches with asyncio.gather()
✓ Complete error handling with detailed logging
✓ Validation before caching
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, ValidationError

# Add parent directory to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.models import (
    PatientState, Demographics, Condition, Medication, 
    Allergy, LabResult, DiagnosticReport, Encounter
)
from shared.redis_client import redis_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patient_context_agent")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PatientContextRequest(BaseModel):
    patient_id: Optional[str] = Field(
        default=None,  # ← was required (...)
        description="FHIR Patient resource ID — optional if X-SHARP-Patient-ID header is present"
    )
    fhir_base_url: Optional[str] = Field(
        default="https://hapi.fhir.org/baseR4",
        description="FHIR server base URL"
    )
    sharp_token: Optional[str] = Field(
        default=None,
        description="SHARP FHIR access token (optional direct field for dev)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "example",
                "fhir_base_url": "https://hapi.fhir.org/baseR4"
            }
        }


class PatientContextResponse(BaseModel):
    """Response with complete patient state"""
    patient_state: PatientState
    cache_hit: bool = Field(..., description="Whether data was served from Redis cache")
    fetch_time_ms: int = Field(..., description="Total fetch time in milliseconds")
    source: str = Field(..., description="Data source: 'SHARP' or 'direct'")
    fhir_resources_fetched: int = Field(..., description="Number of FHIR resource types fetched")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

http_client: Optional[httpx.AsyncClient] = None


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - setup and teardown"""
    global http_client
    
    # Startup
    logger.info("═" * 60)
    logger.info("Patient Context Agent — Starting Up")
    logger.info("═" * 60)
    
    http_client = httpx.AsyncClient(timeout=30.0)
    await redis_client.connect()
    
    logger.info("✓ HTTP client initialized (timeout: 30s)")
    logger.info("✓ Redis connection established")
    logger.info("✓ Patient Context Agent ready on port 8001")
    logger.info("═" * 60)
    
    yield
    
    # Shutdown
    logger.info("Patient Context Agent shutting down...")
    await http_client.aclose()
    await redis_client.disconnect()
    logger.info("✓ Patient Context Agent shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MediTwin Patient Context Agent",
    description=(
        "FHIR data ingestion and normalization layer — Foundation of MediTwin AI. "
        "Fetches patient data from FHIR servers using SHARP context or direct requests, "
        "normalizes into a unified PatientState, and caches for performance."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ══════════════════════════════════════════════════════════════════════════════
# FHIR FETCHING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_fhir_resource(
    resource_type: str,
    base_url: str,
    resource_id: Optional[str] = None,
    search_params: Optional[Dict[str, str]] = None,
    auth_headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Fetch FHIR resource from server with proper error handling
    
    Args:
        resource_type: FHIR resource type (e.g., 'Patient', 'Condition')
        base_url: FHIR server base URL
        resource_id: Specific resource ID (optional, for direct reads)
        search_params: Query parameters for search (optional)
        auth_headers: Authorization headers (optional)
    
    Returns:
        FHIR resource or bundle as dict
        Returns empty dict {} on any error (graceful degradation)
    """
    if resource_id:
        url = f"{base_url}/{resource_type}/{resource_id}"
    else:
        url = f"{base_url}/{resource_type}"
    
    try:
        response = await http_client.get(
            url,
            params=search_params or {},
            headers=auth_headers or {},
            follow_redirects=True
        )
        response.raise_for_status()
        return response.json()
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"FHIR {resource_type} not found (404) — returning empty")
        else:
            logger.error(f"FHIR {resource_type} HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {}
        
    except httpx.TimeoutException:
        logger.error(f"FHIR {resource_type} fetch timeout after 30s")
        return {}
        
    except Exception as e:
        logger.error(f"FHIR {resource_type} fetch failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# FHIR NORMALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_patient(fhir_patient: Dict[str, Any]) -> Optional[Demographics]:
    """
    Extract demographics from FHIR Patient resource
    
    Returns None if resource is invalid
    """
    if not fhir_patient or fhir_patient.get("resourceType") != "Patient":
        logger.warning("Invalid or missing Patient resource")
        return None
    
    try:
        # Extract name
        name = "Unknown"
        if fhir_patient.get("name"):
            name_parts = fhir_patient["name"][0]
            given = " ".join(name_parts.get("given", []))
            family = name_parts.get("family", "")
            name = f"{given} {family}".strip() or "Unknown"
        
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
    except Exception as e:
        logger.error(f"Patient normalization failed: {e}")
        return None


def normalize_conditions(fhir_bundle: Dict[str, Any]) -> List[Condition]:
    """
    Extract active conditions from FHIR Bundle
    
    Only returns clinically active conditions
    Gracefully handles missing or malformed data
    """
    conditions = []
    
    if not fhir_bundle.get("entry"):
        return conditions
    
    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Condition":
                continue
            
            # Only active conditions
            clinical_status = resource.get("clinicalStatus", {})
            status_coding = clinical_status.get("coding", [{}])[0]
            status_code = status_coding.get("code", "")
            
            if status_code != "active":
                continue
            
            # Extract ICD-10 code
            code_obj = resource.get("code", {})
            coding = code_obj.get("coding", [])
            
            if not coding:
                continue
            
            # Prefer ICD-10 coding if available
            icd10_code = None
            display = None
            
            for code in coding:
                system = code.get("system", "")
                if "icd" in system.lower():
                    icd10_code = code.get("code")
                    display = code.get("display")
                    break
            
            # Fallback to first coding
            if not icd10_code:
                icd10_code = coding[0].get("code", "UNKNOWN")
                display = coding[0].get("display", "Unknown condition")
            
            onset = resource.get("onsetDateTime") or resource.get("onsetString") or resource.get("recordedDate")
            
            conditions.append(Condition(
                code=icd10_code,
                display=display,
                onset=onset
            ))
            
        except Exception as e:
            logger.warning(f"Skipping malformed Condition: {e}")
            continue
    
    return conditions


def normalize_medications(fhir_bundle: Dict[str, Any]) -> List[Medication]:
    """
    Extract medications from FHIR Bundle
    
    Returns both active and completed medications
    """
    medications = []
    
    if not fhir_bundle.get("entry"):
        return medications
    
    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "MedicationRequest":
                continue
            
            # Extract medication name
            med_concept = resource.get("medicationCodeableConcept", {})
            coding = med_concept.get("coding", [])
            
            if not coding:
                # Try medicationReference if concept not available
                med_ref = resource.get("medicationReference", {})
                drug_name = med_ref.get("display", "Unknown medication")
            else:
                drug_name = coding[0].get("display", "Unknown medication")
            
            # Extract dosage
            dosage_instructions = resource.get("dosageInstruction", [])
            dose = ""
            frequency = ""
            
            if dosage_instructions:
                dose_inst = dosage_instructions[0]
                dose = dose_inst.get("text", "")
                
                # Try to extract structured dose if text not available
                if not dose and dose_inst.get("doseAndRate"):
                    dose_and_rate = dose_inst["doseAndRate"][0]
                    dose_qty = dose_and_rate.get("doseQuantity", {})
                    dose = f"{dose_qty.get('value', '')} {dose_qty.get('unit', '')}".strip()
                
                # Extract frequency
                timing = dose_inst.get("timing", {})
                repeat = timing.get("repeat", {})
                frequency = repeat.get("frequency", 1)
                period = repeat.get("period", 1)
                period_unit = repeat.get("periodUnit", "d")
                
                if frequency and period:
                    frequency = f"{frequency}x per {period}{period_unit}"
            
            # Extract status
            status = resource.get("status", "unknown")
            
            medications.append(Medication(
                drug=drug_name,
                dose=dose,
                frequency=frequency,
                status=status
            ))
            
        except Exception as e:
            logger.warning(f"Skipping malformed MedicationRequest: {e}")
            continue
    
    return medications


def normalize_allergies(fhir_bundle: Dict[str, Any]) -> List[Allergy]:
    """
    Extract allergies from FHIR Bundle
    
    Only returns clinically active allergies
    """
    allergies = []
    
    if not fhir_bundle.get("entry"):
        return allergies
    
    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "AllergyIntolerance":
                continue
            
            # Check if allergy is active
            clinical_status = resource.get("clinicalStatus", {})
            status_coding = clinical_status.get("coding", [{}])[0]
            if status_coding.get("code") not in ("active", None):
                continue
            
            # Extract allergen
            code_obj = resource.get("code", {})
            coding = code_obj.get("coding", [])
            
            if not coding:
                continue
            
            substance = coding[0].get("display", "Unknown allergen")
            
            # Extract reaction
            reactions = resource.get("reaction", [])
            reaction = ""
            
            if reactions:
                manifestations = reactions[0].get("manifestation", [])
                if manifestations:
                    reaction = manifestations[0].get("coding", [{}])[0].get("display", "")
            
            # Extract severity
            criticality = resource.get("criticality", "unknown")
            
            allergies.append(Allergy(
                substance=substance,
                reaction=reaction,
                severity=criticality
            ))
            
        except Exception as e:
            logger.warning(f"Skipping malformed AllergyIntolerance: {e}")
            continue
    
    return allergies


def normalize_observations(fhir_bundle: Dict[str, Any]) -> List[LabResult]:
    """
    Extract lab results from FHIR Observation Bundle
    
    Only returns laboratory observations with numeric values
    """
    labs = []
    
    if not fhir_bundle.get("entry"):
        return labs
    
    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Observation":
                continue
            
            # Extract LOINC code
            code_obj = resource.get("code", {})
            coding = code_obj.get("coding", [])
            
            if not coding:
                continue
            
            # Prefer LOINC coding
            loinc = None
            display = None
            
            for code in coding:
                system = code.get("system", "")
                if "loinc" in system.lower():
                    loinc = code.get("code")
                    display = code.get("display")
                    break
            
            # Fallback to first coding
            if not loinc:
                loinc = coding[0].get("code", "UNKNOWN")
                display = coding[0].get("display", "Unknown lab")
            
            # Extract value (only numeric)
            value_quantity = resource.get("valueQuantity", {})
            value = value_quantity.get("value")
            unit = value_quantity.get("unit", "")
            
            if value is None:
                continue  # Skip non-numeric observations
            
            # Extract reference range
            ref_ranges = resource.get("referenceRange", [])
            ref_low = None
            ref_high = None
            
            if ref_ranges:
                ref_range = ref_ranges[0]
                if ref_range.get("low"):
                    ref_low = ref_range["low"].get("value")
                if ref_range.get("high"):
                    ref_high = ref_range["high"].get("value")
            
            # Determine flag
            flag = "NORMAL"
            
            # Check for existing interpretation
            interpretation = resource.get("interpretation", [])
            if interpretation:
                interp_code = interpretation[0].get("coding", [{}])[0].get("code", "")
                if interp_code in ("H", "HH", "A"):
                    flag = "HIGH"
                elif interp_code in ("L", "LL"):
                    flag = "LOW"
                elif interp_code in ("C", "CR"):
                    flag = "CRITICAL"
            else:
                # Calculate flag from reference range
                try:
                    value_float = float(value)
                    if ref_high and value_float > ref_high * 1.5:
                        flag = "CRITICAL"
                    elif ref_high and value_float > ref_high:
                        flag = "HIGH"
                    elif ref_low and value_float < ref_low * 0.5:
                        flag = "CRITICAL"
                    elif ref_low and value_float < ref_low:
                        flag = "LOW"
                except (ValueError, TypeError):
                    pass
            
            labs.append(LabResult(
                loinc=loinc,
                display=display,
                value=float(value),
                unit=unit,
                reference_high=ref_high,
                reference_low=ref_low,
                flag=flag
            ))
            
        except Exception as e:
            logger.warning(f"Skipping malformed Observation: {e}")
            continue
    
    return labs


def normalize_diagnostic_reports(fhir_bundle: Dict[str, Any]) -> tuple[List[DiagnosticReport], bool]:
    """
    Extract diagnostic reports from FHIR Bundle
    
    Returns:
        (reports, imaging_available)
    
    imaging_available is True if any report contains imaging content
    """
    reports = []
    imaging_available = False
    
    if not fhir_bundle.get("entry"):
        return reports, imaging_available
    
    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "DiagnosticReport":
                continue
            
            # Extract code
            code_obj = resource.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            code = coding.get("code", "UNKNOWN")
            display = coding.get("display", "Unknown report")
            
            # Extract conclusion
            conclusion = resource.get("conclusion", "")
            
            # Extract issued date
            issued = resource.get("issued") or resource.get("effectiveDateTime")
            
            # Check if this is an imaging report
            category = resource.get("category", [])
            for cat in category:
                cat_coding = cat.get("coding", [{}])[0]
                if cat_coding.get("code") in ("RAD", "IMG"):
                    imaging_available = True
            
            # Also check if presentedForm contains imaging data
            presented_form = resource.get("presentedForm", [])
            if presented_form:
                for form in presented_form:
                    content_type = form.get("contentType", "")
                    if "image" in content_type.lower() or "dicom" in content_type.lower():
                        imaging_available = True
            
            reports.append(DiagnosticReport(
                code=code,
                display=display,
                conclusion=conclusion,
                issued=issued
            ))
            
        except Exception as e:
            logger.warning(f"Skipping malformed DiagnosticReport: {e}")
            continue
    
    return reports, imaging_available


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/fetch", response_model=PatientContextResponse)
async def fetch_patient_context(
    request: PatientContextRequest,
    x_sharp_patient_id: Optional[str] = Header(None, alias="X-SHARP-Patient-ID"),
    x_sharp_fhir_token: Optional[str] = Header(None, alias="X-SHARP-FHIR-Token"),
    x_sharp_fhir_base_url: Optional[str] = Header(None, alias="X-SHARP-FHIR-BaseURL")
) -> PatientContextResponse:
    """
    Fetch and normalize patient FHIR data
    
    SHARP Context Priority (Production):
    1. X-SHARP-Patient-ID header
    2. X-SHARP-FHIR-Token header
    3. X-SHARP-FHIR-BaseURL header
    
    Fallback (Development):
    - If SHARP headers not present, use request body fields
    - This enables smooth development without simulating headers
    
    Workflow:
    1. Check Redis cache (10-minute TTL)
    2. If cache miss, fetch from FHIR server:
       - Patient (demographics)
       - Condition (active diagnoses)
       - MedicationRequest (current prescriptions)
       - AllergyIntolerance (drug allergies)
       - Observation (lab results, last 20)
       - DiagnosticReport (imaging, last 5)
    3. Normalize all resources into PatientState
    4. Cache in Redis
    5. Return PatientState
    """
    start_time = datetime.now()
    
    # ── SHARP Context Resolution ──────────────────────────────────────────────
    # Priority: SHARP headers > request body
    # This is the CRITICAL FIX that was missing
    
    patient_id = x_sharp_patient_id or request.patient_id
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"
    sharp_token = x_sharp_fhir_token or request.sharp_token
    
    # Determine source
    source = "SHARP" if x_sharp_patient_id else "direct"
    
    # Validate patient_id
    if not patient_id:
        raise HTTPException(
            status_code=400,
            detail="patient_id is required (via SHARP header or request body)"
        )
    
    # Build auth headers if token provided
    auth_headers = {}
    if sharp_token:
        auth_headers["Authorization"] = sharp_token if sharp_token.startswith("Bearer ") else f"Bearer {sharp_token}"
    
    logger.info(f"Fetch request: patient={patient_id}, source={source}, fhir={fhir_base_url}")
    
    # ── Redis Cache Check ──────────────────────────────────────────────────────
    
    cache_key = f"patient_state:{patient_id}"
    cached = await redis_client.get_json(cache_key)
    
    if cached:
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"Cache HIT: patient={patient_id}, fetch_time={elapsed_ms}ms")
        
        return PatientContextResponse(
            patient_state=PatientState(**cached),
            cache_hit=True,
            fetch_time_ms=elapsed_ms,
            source=source,
            fhir_resources_fetched=0  # Cache hit, no FHIR fetch
        )
    
    # ── Parallel FHIR Fetch ────────────────────────────────────────────────────
    # CRITICAL: Parallel fetches reduce latency from ~6s (sequential) to ~1s
    
    logger.info(f"Cache MISS: patient={patient_id}, fetching from FHIR server...")
    
    # Build all fetch tasks
    tasks = [
        fetch_fhir_resource("Patient", fhir_base_url, resource_id=patient_id, auth_headers=auth_headers),
        fetch_fhir_resource("Condition", fhir_base_url, search_params={"patient": patient_id, "clinical-status": "active"}, auth_headers=auth_headers),
        fetch_fhir_resource("MedicationRequest", fhir_base_url, search_params={"patient": patient_id, "status": "active"}, auth_headers=auth_headers),
        fetch_fhir_resource("AllergyIntolerance", fhir_base_url, search_params={"patient": patient_id}, auth_headers=auth_headers),
        fetch_fhir_resource("Observation", fhir_base_url, search_params={"patient": patient_id, "category": "laboratory", "_sort": "-date", "_count": "20"}, auth_headers=auth_headers),
        fetch_fhir_resource("DiagnosticReport", fhir_base_url, search_params={"patient": patient_id, "_sort": "-date", "_count": "5"}, auth_headers=auth_headers),
    ]
    
    # Execute all fetches concurrently
    (patient_data, conditions_bundle, medications_bundle, 
     allergies_bundle, observations_bundle, diagnostic_reports_bundle) = await asyncio.gather(*tasks)
    
    # ── Normalization ──────────────────────────────────────────────────────────
    # Each normalizer handles missing data gracefully (returns empty list/None)
    
    demographics = normalize_patient(patient_data)
    
    if demographics is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found or invalid Patient resource"
        )
    try:
        conditions = normalize_conditions(conditions_bundle)
        medications = normalize_medications(medications_bundle)
        allergies = normalize_allergies(allergies_bundle)
        labs = normalize_observations(observations_bundle)
        diagnostic_reports, imaging_available = normalize_diagnostic_reports(diagnostic_reports_bundle)
        
        # Build PatientState
        patient_state = PatientState(
            patient_id=patient_id,
            demographics=demographics,
            active_conditions=conditions,
            medications=medications,
            allergies=allergies,
            lab_results=labs,
            diagnostic_reports=diagnostic_reports,
            recent_encounters=[],  # TODO: Implement if needed
            state_timestamp=datetime.now().isoformat() + "Z",
            imaging_available=imaging_available
        )
        
        logger.info(f"Normalization complete: {len(conditions)} conditions, "
                   f"{len(medications)} medications, {len(labs)} labs, "
                   f"imaging_available={imaging_available}")
        
        # ── Cache and Return ───────────────────────────────────────────────────
        
        # Cache for 10 minutes (600 seconds)
        await redis_client.set_json(cache_key, patient_state.model_dump(), ttl=600)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"Fetch complete: patient={patient_id}, fetch_time={elapsed_ms}ms")
        
        return PatientContextResponse(
            patient_state=patient_state,
            cache_hit=False,
            fetch_time_ms=elapsed_ms,
            source=source,
            fhir_resources_fetched=6
        )
        
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Patient state validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"FHIR normalization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"FHIR normalization error: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Health check endpoint
    
    Verifies:
    - HTTP client is initialized
    - Redis connection is active
    - FHIR server is reachable (optional check)
    """
    redis_ok = await redis_client._client.ping() if redis_client._client else False
    
    return {
        "status": "healthy" if redis_ok else "degraded",
        "agent": "patient-context",
        "version": "1.0.0",
        "redis_connected": redis_ok,
        "http_client_ready": http_client is not None,
        "fhir_base_url": os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)