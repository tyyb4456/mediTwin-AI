import os
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

import httpx


# Add parent directory to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.models import (
    PatientState, Demographics, Condition, Medication, 
    Allergy, LabResult, DiagnosticReport, Encounter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patient_context_agent")


# ══════════════════════════════════════════════════════════════════════════════
# FHIR FETCHING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

import importlib

def _get_http_client():
    """Lazy-fetch the initialized http_client from main.py at call time."""
    main = importlib.import_module("main")
    client = getattr(main, "http_client", None)
    if client is None:
        raise RuntimeError("http_client is not initialized — lifespan may not have run")
    return client

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

    client = _get_http_client()
    if resource_id:
        url = f"{base_url}/{resource_type}/{resource_id}"
    else:
        url = f"{base_url}/{resource_type}"
    
    try:
        response = await client.get(
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


async def _resolve_medication_reference(med_ref: Dict[str, Any], base_url: str, auth_headers: Dict[str, Any]) -> str:
    """
    Resolve a FHIR medicationReference to a human-readable drug name.

    Strategy (in order):
    1. Use inline `display` field — no network call needed
    2. Parse the `reference` field (e.g. "Medication/123") and fetch the
       Medication resource from the FHIR server, then read its name from:
         a. medicationCodeableConcept.coding[].display
         b. medicationCodeableConcept.text
         c. code.coding[].display  (older FHIR layout)
         d. code.text
    3. Fall back to "Unknown medication" if everything above fails
    """
    # 1. Inline display — fastest path, no fetch needed
    display = med_ref.get("display", "").strip()
    if display:
        return display

    # 2. Follow the reference link
    ref_str = med_ref.get("reference", "")
    if not ref_str:
        return "Unknown medication"

    try:
        # Handles both relative ("Medication/123") and absolute URLs
        if ref_str.startswith("http"):
            url = ref_str
        else:
            # e.g. "Medication/abc-123"  →  base_url + "/Medication/abc-123"
            resource_type, resource_id = ref_str.split("/", 1)
            med_resource = await fetch_fhir_resource(
                resource_type, base_url,
                resource_id=resource_id,
                auth_headers=auth_headers
            )
            # 2a. Try medicationCodeableConcept (common in R4)
            for coding in med_resource.get("medicationCodeableConcept", {}).get("coding", []):
                if coding.get("display"):
                    return coding["display"]
            text = med_resource.get("medicationCodeableConcept", {}).get("text", "")
            if text:
                return text
            # 2b. Try top-level code (also common)
            for coding in med_resource.get("code", {}).get("coding", []):
                if coding.get("display"):
                    return coding["display"]
            text = med_resource.get("code", {}).get("text", "")
            if text:
                return text

    except Exception as e:
        logger.warning(f"Could not resolve medicationReference '{ref_str}': {e}")

    return "Unknown medication"


async def normalize_medications(
    fhir_bundle: Dict[str, Any],
    base_url: str = "https://hapi.fhir.org/baseR4",
    auth_headers: Optional[Dict[str, Any]] = None,
) -> List[Medication]:
    """
    Extract medications from FHIR Bundle.

    Returns both active and completed medications.
    Resolves medicationReference → Medication resource when the inline
    display name is missing, so you never get "Unknown medication" again.
    """
    medications = []
    auth_headers = auth_headers or {}

    if not fhir_bundle.get("entry"):
        return medications

    for entry in fhir_bundle.get("entry", []):
        try:
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "MedicationRequest":
                continue

            # ── Resolve drug name ────────────────────────────────────────────
            # DEBUG: log raw medication fields so we can see what FHIR sends
            logger.info(
                f"[MED DEBUG] medicationCodeableConcept={resource.get('medicationCodeableConcept')} | "
                f"medicationReference={resource.get('medicationReference')} | "
                f"contained={[c.get('resourceType') for c in resource.get('contained', [])]}"
            )

            # Priority 1 — medicationCodeableConcept
            # Check coding[].display first, then .text (HAPI often omits coding entirely)
            med_concept = resource.get("medicationCodeableConcept", {})
            coding = med_concept.get("coding", [])
            concept_name = (
                next((c.get("display") for c in coding if c.get("display")), None)
                or med_concept.get("text", "")
            )
            if concept_name:
                drug_name = concept_name

            # Priority 2 — contained Medication resource (embedded inline by many FHIR servers)
            elif resource.get("contained"):
                contained_med = next(
                    (c for c in resource["contained"] if c.get("resourceType") == "Medication"),
                    None
                )
                if contained_med:
                    drug_name = ""
                    for c in contained_med.get("code", {}).get("coding", []):
                        if c.get("display"):
                            drug_name = c["display"]
                            break
                    if not drug_name:
                        drug_name = contained_med.get("code", {}).get("text", "")
                    if not drug_name:
                        for ingredient in contained_med.get("ingredient", []):
                            for c in ingredient.get("itemCodeableConcept", {}).get("coding", []):
                                if c.get("display"):
                                    drug_name = c["display"]
                                    break
                            if drug_name:
                                break
                    drug_name = drug_name or "Unknown medication"
                else:
                    drug_name = "Unknown medication"

            # Priority 3 — external medicationReference (requires a FHIR fetch)
            else:
                med_ref = resource.get("medicationReference", {})
                drug_name = await _resolve_medication_reference(med_ref, base_url, auth_headers)

            logger.info(f"[MED DEBUG] Resolved drug name → '{drug_name}'")
            
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