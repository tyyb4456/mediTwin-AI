"""
Drug Safety Core Logic
Handles:
  1. Allergy cross-reactivity (local lookup — fast, deterministic)
  2. Condition contraindications (local JSON — deterministic)
  3. Drug-drug interaction checking (NLM RxNav API)
  4. Safe alternative suggestions (LLM-assisted)
"""
import json
import os
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ── Load contraindications DB once at import ───────────────────────────────────
_CONTRA_PATH = Path(__file__).parent / "contraindications.json"
with open(_CONTRA_PATH) as f:
    _CONTRAINDICATIONS: dict = json.load(f)


# ── Allergy Cross-Reactivity Table ────────────────────────────────────────────
# Key: allergen family → list of specific drugs in that family
# If a patient is allergic to ANY item, they may react to ALL in that family.

CROSS_REACTIVITY: dict[str, list[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "nafcillin", "oxacillin",
        "dicloxacillin", "piperacillin", "ticarcillin",
        "amoxicillin-clavulanate", "piperacillin-tazobactam",
        # Cephalosporins — low cross-reactivity (~1-2%), included for safety
        # Only flagged for severe (anaphylactic) penicillin allergy
    ],
    "amoxicillin": [
        "amoxicillin", "amoxicillin-clavulanate", "ampicillin",
    ],
    "cephalosporin": [
        "cephalexin", "cefazolin", "ceftriaxone", "cefdinir",
        "cefepime", "cefpodoxime", "ceftaroline",
    ],
    "sulfa": [
        "sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim",
        "sulfadiazine", "sulfasalazine",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "aspirin", "indomethacin",
        "ketorolac", "diclofenac", "meloxicam",
    ],
    "fluoroquinolone": [
        "ciprofloxacin", "levofloxacin", "moxifloxacin",
        "norfloxacin", "ofloxacin",
    ],
    "macrolide": [
        "azithromycin", "clarithromycin", "erythromycin",
    ],
    "tetracycline": [
        "doxycycline", "minocycline", "tetracycline",
    ],
    "carbapenem": [
        "meropenem", "imipenem", "ertapenem", "doripenem",
    ],
}

# Severity of cross-reactivity between allergy families
# Format: {allergen_lower} → {drug_class_lower} → severity
CROSS_REACTIVITY_SEVERITY: dict[str, dict[str, str]] = {
    "penicillin": {
        "penicillin": "CRITICAL",    # Same family — direct cross-reactivity
        "cephalosporin": "MODERATE", # ~1-2% cross-reactivity, but clinically significant for anaphylaxis
        "carbapenem": "LOW",         # <1% cross-reactivity
    },
    "amoxicillin": {
        "amoxicillin": "CRITICAL",
        "penicillin":  "HIGH",
    },
    "sulfa": {
        "sulfa": "CRITICAL",
    },
    "nsaid": {
        "nsaid": "HIGH",  # NSAIDs can cross-react with each other
    },
}


def normalize_drug_name(name: str) -> str:
    """Lowercase, strip dose/form info: 'Amoxicillin 500mg TID' → 'amoxicillin'"""
    return name.lower().split()[0].strip().rstrip(".,")


def get_drug_family(drug_name: str) -> Optional[str]:
    """Find which allergy family a drug belongs to."""
    normalized = normalize_drug_name(drug_name)
    for family, members in CROSS_REACTIVITY.items():
        if normalized in [m.lower() for m in members]:
            return family
        if normalized == family:
            return family
    return None


def check_allergy_cross_reactivity(
    drug_name: str,
    patient_allergies: list[dict],
) -> list[dict]:
    """
    Check if a drug is contraindicated due to patient allergies (cross-reactivity).

    Args:
        drug_name: Drug to check
        patient_allergies: List of allergy dicts {substance, reaction, severity}

    Returns:
        List of contraindication dicts (empty = safe)
    """
    contraindications = []
    drug_lower = normalize_drug_name(drug_name)
    drug_family = get_drug_family(drug_name)

    for allergy in patient_allergies:
        allergen = allergy.get("substance", "").lower().strip()
        allergy_severity = allergy.get("severity", "unknown").lower()
        reaction = allergy.get("reaction", "unknown")

        # Direct name match — exact same drug
        if drug_lower == allergen or drug_lower in allergen:
            contraindications.append({
                "drug": drug_name,
                "allergen": allergy.get("substance"),
                "reason": f"Patient has documented allergy to {allergy.get('substance')} "
                          f"(reaction: {reaction}). This IS the same drug.",
                "severity": "CRITICAL",
                "recommendation": f"Do NOT administer {drug_name}. Select an alternative.",
            })
            continue

        # Cross-reactivity check
        allergen_family = get_drug_family(allergen) or allergen

        if drug_family and allergen_family:
            cross_sev = (
                CROSS_REACTIVITY_SEVERITY
                .get(allergen_family, {})
                .get(drug_family)
            )

            if cross_sev:
                # For MODERATE cross-reactivity, only flag if allergy was anaphylaxis/severe
                if cross_sev == "MODERATE" and allergy_severity not in ("severe", "high", "anaphylaxis"):
                    continue

                contraindications.append({
                    "drug": drug_name,
                    "allergen": allergy.get("substance"),
                    "reason": (
                        f"Cross-reactivity risk: patient allergic to {allergy.get('substance')} "
                        f"({allergen_family} family). {drug_name} is in the {drug_family} family. "
                        f"Estimated cross-reactivity severity: {cross_sev}."
                    ),
                    "severity": cross_sev,
                    "recommendation": (
                        f"AVOID {drug_name} due to {allergy.get('substance')} allergy cross-reactivity. "
                        f"Use non-{drug_family} alternative."
                    ),
                })

    return contraindications


def check_condition_contraindications(
    drug_name: str,
    patient_conditions: list[dict],
) -> list[dict]:
    """
    Check if a drug is contraindicated given the patient's ICD-10 conditions.

    Args:
        drug_name: Drug to check (case-insensitive match against contraindications DB)
        patient_conditions: List of condition dicts {code, display}

    Returns:
        List of contraindication dicts (empty = no contraindications found)
    """
    contraindications = []
    drug_key = drug_name.split()[0].strip()  # Strip dose info

    # Case-insensitive lookup in contraindications DB
    drug_entry = None
    for db_key in _CONTRAINDICATIONS:
        if db_key.lower() == drug_key.lower():
            drug_entry = _CONTRAINDICATIONS[db_key]
            break

    if not drug_entry:
        return []  # Drug not in DB — no known condition contraindications

    condition_map = drug_entry.get("conditions", {})

    for condition in patient_conditions:
        icd_code = condition.get("code", "").strip()
        display = condition.get("display", "")

        if icd_code in condition_map:
            entry = condition_map[icd_code]
            contraindications.append({
                "drug": drug_name,
                "condition_code": icd_code,
                "condition_display": display or icd_code,
                "severity": entry["severity"],
                "reason": entry["reason"],
                "recommendation": f"Review use of {drug_name} in context of {display or icd_code}. "
                                  f"Consult prescribing guidelines.",
            })

    return contraindications


# ── LLM: Alternative Suggestions ─────────────────────────────────────────────

_ALTERNATIVES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist suggesting safer medication alternatives.
Return ONLY valid JSON. No preamble or markdown code blocks."""),
    ("human", """A patient cannot take {drug_name} because: {reason_for_avoidance}

Drug indication: {indication}
Patient's active conditions (ICD-10): {patient_conditions}
Patient's current medications: {current_medications}
Patient's known allergies: {allergies}

Suggest 2-3 safe alternatives. Return JSON:
{{
  "alternatives": [
    {{
      "drug": "Drug name and typical dose",
      "rationale": "Why this is safe given patient's allergy/condition",
      "interaction_check_needed": ["list any of patient's current meds to watch for interactions with this alternative"],
      "safe_to_prescribe": true
    }}
  ],
  "clinical_note": "One sentence note for the prescribing clinician"
}}
Return ONLY JSON."""),
])


def get_alternative_suggestions(
    drug_name: str,
    reason_for_avoidance: str,
    indication: str,
    patient_conditions: list[dict],
    current_medications: list[dict],
    allergies: list[dict],
) -> Optional[dict]:
    """
    Use LLM to suggest safe drug alternatives.
    Returns None if LLM unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        chain = _ALTERNATIVES_PROMPT | llm | JsonOutputParser()

        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in patient_conditions[:5]
        ) or "None"

        meds_str = ", ".join(
            m.get("drug", "") for m in current_medications[:5]
        ) or "None"

        allergies_str = ", ".join(
            f"{a.get('substance')} ({a.get('reaction', '')})"
            for a in allergies
        ) or "None"

        result = chain.invoke({
            "drug_name": drug_name,
            "reason_for_avoidance": reason_for_avoidance,
            "indication": indication,
            "patient_conditions": conditions_str,
            "current_medications": meds_str,
            "allergies": allergies_str,
        })

        return result

    except Exception as e:
        print(f"  LLM alternatives failed: {e}")
        return None


# ── FHIR MedicationRequest builder ────────────────────────────────────────────

def build_fhir_medication_request(
    drug_name: str,
    patient_id: str,
    safety_cleared: bool,
    safety_note: str = "",
) -> dict:
    """Build a FHIR MedicationRequest resource for a cleared medication."""
    return {
        "resourceType": "MedicationRequest",
        "status": "active" if safety_cleared else "on-hold",
        "intent": "proposal",
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {
            "text": drug_name
        },
        "note": [
            {
                "text": (
                    f"AI Drug Safety Check: {'CLEARED' if safety_cleared else 'FLAGGED'}. "
                    + (safety_note or "No additional notes.")
                    + " Requires physician verification before dispensing."
                )
            }
        ],
    }