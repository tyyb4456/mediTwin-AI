"""
Drug Safety Core Logic — v2.0
Handles:
  1. Allergy cross-reactivity (local lookup — fast, deterministic)
  2. Condition contraindications (local JSON — deterministic)
  3. Drug-drug interaction checking (NLM RxNav API)
  4. Clinical enrichment via LLM with structured Pydantic output
  5. Safe alternative suggestions (LLM with structured output)
  6. Patient-specific risk profiling (LLM with structured output)

LLM Strategy:
  - All LLM calls use llm.with_structured_output(PydanticModel)
  - No free-form JSON parsing — structured output guarantees schema conformance
  - LLM is ADDITIVE to deterministic rules — rules run first, LLM enriches
"""
import json
import os
from pathlib import Path
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


# ── Load contraindications DB once at import ───────────────────────────────────
_CONTRA_PATH = Path(__file__).parent / "contraindications.json"
with open(_CONTRA_PATH) as f:
    _CONTRAINDICATIONS: dict = json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models for structured LLM output
# ══════════════════════════════════════════════════════════════════════════════

class InteractionSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"
    UNKNOWN  = "UNKNOWN"


class AlternativeDrug(BaseModel):
    drug: str = Field(description="Drug name with typical dose, e.g. 'Azithromycin 500mg PO daily x5 days'")
    rationale: str = Field(description="Clinical rationale — why safe given patient context, max 2 sentences")
    drug_class: str = Field(description="Drug class, e.g. 'Macrolide antibiotic'")
    interaction_check_needed: list[str] = Field(
        default_factory=list,
        description="Current patient medications that need monitoring with this alternative"
    )
    safe_to_prescribe: bool = Field(description="True if no known contraindications given patient context")
    cautions: Optional[str] = Field(default=None, description="Any specific monitoring or caution notes")


class AlternativeSuggestionsOutput(BaseModel):
    """Structured output for suggest_alternatives LLM call."""
    alternatives: list[AlternativeDrug] = Field(description="2-3 safe alternative drugs ranked by preference")
    clinical_note: str = Field(description="One-sentence note for the prescribing clinician summarizing the switch")
    urgency: str = Field(
        description="How urgently an alternative is needed: IMMEDIATE | BEFORE_NEXT_DOSE | ROUTINE"
    )


class InteractionEnrichment(BaseModel):
    """LLM-enriched clinical context for a single drug interaction."""
    mechanism: str = Field(description="Pharmacological mechanism of the interaction in 1-2 sentences")
    clinical_significance: str = Field(
        description="What actually happens to the patient: specific effects, onset, severity context"
    )
    monitoring_parameters: list[str] = Field(
        description="Specific lab values or clinical signs to monitor, e.g. 'INR every 48h', 'serum creatinine'"
    )
    management_strategy: str = Field(
        description="Concrete clinical action: dose adjustment, timing separation, alternative, or discontinue"
    )
    time_to_onset: str = Field(description="When effect typically manifests: e.g. 'Within 24-48h', '3-5 days'")


class InteractionEnrichmentBatch(BaseModel):
    """Batch enrichment for multiple interactions."""
    enriched_interactions: list[InteractionEnrichment] = Field(
        description="One enrichment per interaction, in the same order as input"
    )
    overall_risk_narrative: str = Field(
        description="2-3 sentence summary of the combined interaction risk for this patient"
    )


class PatientRiskProfile(BaseModel):
    """Patient-specific drug risk assessment considering full clinical picture."""
    overall_risk_level: str = Field(
        description="CRITICAL | HIGH | MODERATE | LOW | MINIMAL"
    )
    primary_risk_factors: list[str] = Field(
        description="Top 3 patient-specific factors driving the risk (e.g. 'Penicillin allergy + Amoxicillin proposed')"
    )
    safe_to_proceed: bool = Field(
        description="True only if no CRITICAL or HIGH contraindications exist"
    )
    recommended_action: str = Field(
        description="PRESCRIBE | PRESCRIBE_WITH_MONITORING | SWITCH_DRUG | CONSULT_PHARMACIST | DO_NOT_PRESCRIBE"
    )
    clinical_summary: str = Field(
        description="3-4 sentence clinical summary synthesizing all safety findings for this specific patient"
    )
    special_populations_flag: Optional[str] = Field(
        default=None,
        description="If applicable: ELDERLY | PEDIATRIC | RENAL_IMPAIRMENT | HEPATIC_IMPAIRMENT | PREGNANCY"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Allergy Cross-Reactivity Table (expanded)
# ══════════════════════════════════════════════════════════════════════════════

CROSS_REACTIVITY: dict[str, list[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "nafcillin", "oxacillin",
        "dicloxacillin", "piperacillin", "ticarcillin",
        "amoxicillin-clavulanate", "piperacillin-tazobactam",
        "penicillin", "penicillin-v", "penicillin-g",
    ],
    "amoxicillin": [
        "amoxicillin", "amoxicillin-clavulanate", "ampicillin",
    ],
    "cephalosporin": [
        "cephalexin", "cefazolin", "ceftriaxone", "cefdinir",
        "cefepime", "cefpodoxime", "ceftaroline", "cefuroxime",
        "cefadroxil", "cephalothin",
    ],
    "carbapenem": [
        "meropenem", "imipenem", "ertapenem", "doripenem",
    ],
    "sulfa": [
        "sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim",
        "sulfadiazine", "sulfasalazine", "cotrimoxazole",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "aspirin", "indomethacin",
        "ketorolac", "diclofenac", "meloxicam", "celecoxib",
        "piroxicam", "etodolac",
    ],
    "fluoroquinolone": [
        "ciprofloxacin", "levofloxacin", "moxifloxacin",
        "norfloxacin", "ofloxacin", "gemifloxacin",
    ],
    "macrolide": [
        "azithromycin", "clarithromycin", "erythromycin", "roxithromycin",
    ],
    "tetracycline": [
        "doxycycline", "minocycline", "tetracycline", "demeclocycline",
    ],
    "aminoglycoside": [
        "gentamicin", "tobramycin", "amikacin", "streptomycin", "neomycin",
    ],
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
    "ace_inhibitor": [
        "lisinopril", "enalapril", "ramipril", "captopril",
        "perindopril", "benazepril", "quinapril",
    ],
    "arb": [
        "losartan", "valsartan", "irbesartan", "candesartan",
        "olmesartan", "telmisartan",
    ],
    "opioid": [
        "morphine", "codeine", "oxycodone", "hydrocodone",
        "fentanyl", "tramadol", "hydromorphone",
    ],
    "benzodiazepine": [
        "diazepam", "lorazepam", "alprazolam", "clonazepam",
        "midazolam", "temazepam", "oxazepam",
    ],
    "contrast_dye": [
        "iodinated-contrast", "gadolinium", "iopromide",
    ],
}

# Severity of cross-reactivity between allergy families
CROSS_REACTIVITY_SEVERITY: dict[str, dict[str, str]] = {
    "penicillin": {
        "penicillin":   "CRITICAL",
        "amoxicillin":  "CRITICAL",
        "cephalosporin":"MODERATE",  # ~1-2%, only flag for severe allergy
        "carbapenem":   "LOW",
    },
    "amoxicillin": {
        "amoxicillin": "CRITICAL",
        "penicillin":  "HIGH",
    },
    "cephalosporin": {
        "cephalosporin": "CRITICAL",
        "penicillin":    "LOW",
    },
    "sulfa": {
        "sulfa": "CRITICAL",
    },
    "nsaid": {
        "nsaid": "HIGH",
    },
    "fluoroquinolone": {
        "fluoroquinolone": "HIGH",
    },
    "macrolide": {
        "macrolide": "MODERATE",
    },
    "tetracycline": {
        "tetracycline": "MODERATE",
    },
    "aminoglycoside": {
        "aminoglycoside": "HIGH",
    },
    "opioid": {
        "opioid": "MODERATE",
    },
    "benzodiazepine": {
        "benzodiazepine": "MODERATE",
    },
    "statin": {
        "statin": "LOW",  # Statin myopathy cross-reactivity is low
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
    Deterministic — no LLM involved.
    """
    contraindications = []
    drug_lower = normalize_drug_name(drug_name)
    drug_family = get_drug_family(drug_name)

    for allergy in patient_allergies:
        allergen = allergy.get("substance", "").lower().strip()
        allergy_severity = allergy.get("severity", "unknown").lower()
        reaction = allergy.get("reaction", "unknown")

        # Direct name match
        if drug_lower == allergen or drug_lower in allergen:
            contraindications.append({
                "drug": drug_name,
                "allergen": allergy.get("substance"),
                "reason": (
                    f"Patient has documented allergy to {allergy.get('substance')} "
                    f"(reaction: {reaction}). This IS the same drug or a direct match."
                ),
                "severity": "CRITICAL",
                "recommendation": f"Do NOT administer {drug_name}. Select a non-cross-reactive alternative.",
                "type": "direct_allergy",
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
                # MODERATE cross-reactivity: only flag for severe/anaphylactic allergy
                if cross_sev == "MODERATE" and allergy_severity not in (
                    "severe", "high", "anaphylaxis", "anaphylactic"
                ):
                    continue
                # LOW cross-reactivity: only flag for anaphylaxis history
                if cross_sev == "LOW" and allergy_severity not in ("anaphylaxis", "anaphylactic"):
                    continue

                contraindications.append({
                    "drug": drug_name,
                    "allergen": allergy.get("substance"),
                    "reason": (
                        f"Cross-reactivity risk: patient allergic to {allergy.get('substance')} "
                        f"({allergen_family} family). {drug_name} is in the {drug_family} family. "
                        f"Cross-reactivity severity: {cross_sev}."
                    ),
                    "severity": cross_sev,
                    "recommendation": (
                        f"AVOID {drug_name} — {allergy.get('substance')} allergy cross-reactivity. "
                        f"Use a non-{drug_family} alternative."
                    ),
                    "type": "cross_reactivity",
                })

    return contraindications


def check_condition_contraindications(
    drug_name: str,
    patient_conditions: list[dict],
) -> list[dict]:
    """
    Check if a drug is contraindicated given the patient's ICD-10 conditions.
    Deterministic — no LLM involved.
    """
    contraindications = []
    drug_key = drug_name.split()[0].strip()

    drug_entry = None
    for db_key in _CONTRAINDICATIONS:
        if db_key.lower() == drug_key.lower():
            drug_entry = _CONTRAINDICATIONS[db_key]
            break

    if not drug_entry:
        return []

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
                "recommendation": (
                    f"Review use of {drug_name} in context of {display or icd_code}. "
                    f"Consult prescribing guidelines."
                ),
                "type": "condition_contraindication",
            })

    return contraindications


# ── LLM: Enrich interaction list with clinical context ─────────────────────

_INTERACTION_ENRICHMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior clinical pharmacist providing structured drug interaction analysis. "
     "Be precise, evidence-based, and clinically actionable. Do not include disclaimers."),
    ("human",
     """Patient context:
- Age: {age}, Gender: {gender}
- Active conditions: {conditions}
- Current medications: {current_medications}

Drug interactions detected by RxNav:
{interactions_json}

For each interaction listed above (in the same order), provide clinical enrichment.
Also write an overall risk narrative for this specific patient."""),
])


async def enrich_interactions_with_llm(
    interactions: list[dict],
    patient_state: dict,
) -> Optional[InteractionEnrichmentBatch]:
    """
    Use LLM with structured output to enrich raw RxNav interactions with
    clinical context: mechanism, monitoring parameters, management strategy.

    Returns None if LLM unavailable or interactions list is empty.
    """
    if not interactions:
        return None

    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0
        )
    if not llm:
        return None

    try:
        structured_llm = llm.with_structured_output(InteractionEnrichmentBatch)

        demographics = patient_state.get("demographics", {})
        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in patient_state.get("active_conditions", [])[:6]
        ) or "None documented"
        meds_str = ", ".join(
            m.get("drug", "") for m in patient_state.get("medications", [])[:6]
        ) or "None documented"

        # Build a clean summary for the LLM (avoid sending full RxNav noise)
        interactions_summary = [
            {
                "drug_a": i.get("drug_a"),
                "drug_b": i.get("drug_b"),
                "severity": i.get("severity"),
                "rxnav_description": i.get("description", ""),
            }
            for i in interactions
        ]

        chain = _INTERACTION_ENRICHMENT_PROMPT | structured_llm

        result = await chain.ainvoke({
            "age": demographics.get("age", "unknown"),
            "gender": demographics.get("gender", "unknown"),
            "conditions": conditions_str,
            "current_medications": meds_str,
            "interactions_json": json.dumps(interactions_summary, indent=2),
        })

        return result

    except Exception as e:
        print(f"  ⚠️  Interaction enrichment LLM failed: {e}")
        return None


# ── LLM: Patient-specific overall risk profile ─────────────────────────────

_RISK_PROFILE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a clinical pharmacist generating a structured patient-specific drug safety assessment. "
     "Be specific, use the actual drug names and patient data provided. Do not be generic."),
    ("human",
     """Patient:
- Demographics: Age {age}, Gender {gender}
- Active conditions: {conditions}
- Current medications: {current_medications}
- Allergies: {allergies}
- Renal function hint: {renal_hint}

Drug safety findings:
- Proposed medications: {proposed_meds}
- Contraindications found ({contra_count}): {contraindications_json}
- Drug interactions found ({interaction_count}): {interactions_json}
- FDA black box warnings: {fda_warnings_json}
- Overall safety status: {safety_status}

Generate a patient-specific risk profile and clinical summary."""),
])


async def generate_patient_risk_profile(
    patient_state: dict,
    proposed_medications: list[str],
    contraindications: list[dict],
    interactions: list[dict],
    fda_warnings: dict,
    safety_status: str,
) -> Optional[PatientRiskProfile]:
    """
    Generate a patient-specific risk profile using structured LLM output.
    This synthesizes ALL safety signals into one coherent clinical assessment.
    """
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0
        )
    if not llm:
        return None

    try:
        structured_llm = llm.with_structured_output(PatientRiskProfile)
        chain = _RISK_PROFILE_PROMPT | structured_llm

        demographics = patient_state.get("demographics", {})

        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in patient_state.get("active_conditions", [])[:6]
        ) or "None"
        meds_str = ", ".join(
            m.get("drug", "") for m in patient_state.get("medications", [])[:6]
        ) or "None"
        allergies_str = ", ".join(
            f"{a.get('substance')} → {a.get('reaction', '?')} ({a.get('severity', '?')})"
            for a in patient_state.get("allergies", [])
        ) or "NKDA"

        # Detect renal impairment from conditions
        renal_conditions = [
            c for c in patient_state.get("active_conditions", [])
            if c.get("code", "").startswith("N18") or c.get("code", "").startswith("N17")
        ]
        renal_hint = (
            f"Renal impairment noted: {renal_conditions[0].get('display')}"
            if renal_conditions else "No renal impairment conditions documented"
        )

        result = await chain.ainvoke({
            "age": demographics.get("age", "unknown"),
            "gender": demographics.get("gender", "unknown"),
            "conditions": conditions_str,
            "current_medications": meds_str,
            "allergies": allergies_str,
            "renal_hint": renal_hint,
            "proposed_meds": ", ".join(proposed_medications),
            "contra_count": len(contraindications),
            "contraindications_json": json.dumps(
                [{"drug": c["drug"], "severity": c["severity"], "reason": c["reason"]}
                 for c in contraindications[:5]], indent=2
            ),
            "interaction_count": len(interactions),
            "interactions_json": json.dumps(
                [{"drugs": f"{i.get('drug_a')} + {i.get('drug_b')}", "severity": i.get("severity")}
                 for i in interactions[:5]], indent=2
            ),
            "fda_warnings_json": json.dumps(
                {k: v[:2] for k, v in list(fda_warnings.items())[:3] if v}  # top 3 drugs, 2 warnings each
            ),
            "safety_status": safety_status,
        })

        return result

    except Exception as e:
        print(f"  ⚠️  Risk profile LLM failed: {e}")
        return None


# ── LLM: Alternative suggestions (structured output) ──────────────────────

_ALTERNATIVES_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior clinical pharmacist recommending medication alternatives. "
     "Suggestions must be specific, avoid the patient's known allergens, and be practical."),
    ("human",
     """A patient cannot take {drug_name} because: {reason_for_avoidance}

Drug indication: {indication}

Patient context:
- Active ICD-10 conditions: {patient_conditions}
- Current medications: {current_medications}
- Known allergies: {allergies}

Suggest 2-3 safe alternatives ranked by clinical preference. 
Focus on drugs in different allergy families.
Include urgency of switching."""),
])


async def get_alternative_suggestions_async(
    drug_name: str,
    reason_for_avoidance: str,
    indication: str,
    patient_conditions: list[dict],
    current_medications: list[dict],
    allergies: list[dict],
) -> Optional[AlternativeSuggestionsOutput]:
    """
    Use LLM with structured output to suggest safe alternatives.
    Async version — safe to call from FastAPI/MCP handlers.
    """
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0
        )
    if not llm:
        return None

    try:
        structured_llm = llm.with_structured_output(AlternativeSuggestionsOutput)
        chain = _ALTERNATIVES_PROMPT | structured_llm

        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in patient_conditions[:5]
        ) or "None"
        meds_str = ", ".join(m.get("drug", "") for m in current_medications[:5]) or "None"
        allergies_str = ", ".join(
            f"{a.get('substance')} (reaction: {a.get('reaction', '?')}, severity: {a.get('severity', '?')})"
            for a in allergies
        ) or "NKDA (No Known Drug Allergies)"

        result = await chain.ainvoke({
            "drug_name": drug_name,
            "reason_for_avoidance": reason_for_avoidance,
            "indication": indication,
            "patient_conditions": conditions_str,
            "current_medications": meds_str,
            "allergies": allergies_str,
        })

        return result

    except Exception as e:
        print(f"  ⚠️  LLM alternatives failed: {e}")
        return None


# Keep sync version for backward compat — now wraps async
def get_alternative_suggestions(
    drug_name: str,
    reason_for_avoidance: str,
    indication: str,
    patient_conditions: list[dict],
    current_medications: list[dict],
    allergies: list[dict],
) -> Optional[dict]:
    """Sync wrapper — use get_alternative_suggestions_async in async contexts."""
    import asyncio
    try:
        result = asyncio.get_event_loop().run_until_complete(
            get_alternative_suggestions_async(
                drug_name, reason_for_avoidance, indication,
                patient_conditions, current_medications, allergies,
            )
        )
        return result.model_dump() if result else None
    except Exception:
        return None


# ── FHIR MedicationRequest builder ────────────────────────────────────────────

def build_fhir_medication_request(
    drug_name: str,
    patient_id: str,
    safety_cleared: bool,
    safety_note: str = "",
    risk_level: str = "LOW",
) -> dict:
    """Build a FHIR R4 MedicationRequest resource for a cleared medication."""
    status = "active" if safety_cleared else "on-hold"
    priority = "urgent" if risk_level in ("CRITICAL", "HIGH") else "routine"

    return {
        "resourceType": "MedicationRequest",
        "status": status,
        "intent": "proposal",
        "priority": priority,
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {
            "text": drug_name
        },
        "extension": [
            {
                "url": "https://meditwin.ai/fhir/StructureDefinition/ai-safety-check",
                "valueString": f"CLEARED" if safety_cleared else "FLAGGED"
            }
        ],
        "note": [
            {
                "text": (
                    f"MediTwin Drug Safety Check (v2.0): "
                    f"{'CLEARED' if safety_cleared else 'FLAGGED — DO NOT DISPENSE'}. "
                    + (safety_note or "No additional safety notes.")
                    + " ⚠️ AI-generated — requires physician verification before dispensing."
                )
            }
        ],
    }