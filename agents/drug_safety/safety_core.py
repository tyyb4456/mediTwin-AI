"""
Drug Safety Core Logic — v3.0
Clinical standards upgrade over v2.0:

  NEW in v3.0:
  ─────────────────────────────────────────────────────────────────────────────
  1. INTERACTION_SEVERITY_OVERRIDE table
     FDA label checks return MODERATE for everything by default. This table
     upgrades known clinically significant pairs to HIGH or CRITICAL based on
     published guidelines (AHA, ATS/IDSA, ASHP).

  2. assess_critical_labs()
     Reads patient lab results, flags critical values (WBC, Cr, K, INR, etc.),
     and produces a structured ClinicalLabContext that is injected into:
       - The LLM risk profile prompt (raises urgency when e.g. WBC=18.4 suggests sepsis)
       - The alternatives prompt (steers away from renally-cleared drugs if Cr is high)
       - The FHIR note for each drug (adds specific clinical context)

  3. Proactive alternatives
     generate_proactive_alternatives() is called for every flagged drug at the
     end of the pipeline. It passes the lab context and all other patient data
     so the LLM can suggest clinically appropriate substitutes immediately.

  4. Drug-specific FHIR notes
     build_fhir_medication_request() now accepts a specific_reason parameter so
     each flagged drug carries its own contraindication reason in the FHIR note.

  Existing logic unchanged:
  ─────────────────────────────────────────────────────────────────────────────
  - Allergy cross-reactivity (deterministic)
  - Condition contraindications (local JSON)
  - LLM interaction enrichment (InteractionEnrichmentBatch)
  - LLM patient risk profile (PatientRiskProfile)
  - All Pydantic models for structured LLM output
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
# NEW v3.0: Interaction severity override table
# ══════════════════════════════════════════════════════════════════════════════
# FDA label checks return MODERATE for all label mentions by default.
# This table upgrades known dangerous pairs to their true clinical severity.
# Sources: ASHP Drug Interaction Handbook, AHA anticoagulation guidelines,
#          ATS/IDSA pneumonia guidelines, BNF interaction appendix.
#
# Key format: frozenset({drug_a_base, drug_b_base}) — case-insensitive base names.
# "Base name" = first word, lowercased, stripped of dose: "warfarin 5mg" → "warfarin"

INTERACTION_SEVERITY_OVERRIDE: dict[frozenset, dict] = {
    # Anticoagulant + NSAID — major bleeding risk
    frozenset({"warfarin", "ibuprofen"}):   {"severity": "HIGH",   "reason": "Warfarin + NSAID: INR potentiation + GI mucosal damage — major bleeding risk per ASHP/AHA"},
    frozenset({"warfarin", "naproxen"}):    {"severity": "HIGH",   "reason": "Warfarin + NSAID: INR potentiation + GI bleeding risk"},
    frozenset({"warfarin", "aspirin"}):     {"severity": "HIGH",   "reason": "Warfarin + Aspirin: additive anticoagulation + GI bleeding — HIGH risk"},
    frozenset({"warfarin", "diclofenac"}):  {"severity": "HIGH",   "reason": "Warfarin + NSAID: INR elevation + GI bleeding risk"},
    frozenset({"warfarin", "ketorolac"}):   {"severity": "CRITICAL","reason": "Warfarin + Ketorolac: CRITICAL bleeding risk — avoid combination"},

    # Anticoagulant + antiplatelet
    frozenset({"warfarin", "clopidogrel"}): {"severity": "HIGH",   "reason": "Warfarin + Clopidogrel: dual antithrombotic — major bleeding risk"},

    # Anticoagulant + macrolide (CYP2C9 inhibition → INR spike)
    frozenset({"warfarin", "azithromycin"}):     {"severity": "HIGH", "reason": "Warfarin + Azithromycin: CYP2C9 inhibition raises INR — monitor closely"},
    frozenset({"warfarin", "clarithromycin"}):   {"severity": "HIGH", "reason": "Warfarin + Clarithromycin: strong CYP3A4/2C9 inhibition — significant INR elevation"},
    frozenset({"warfarin", "metronidazole"}):    {"severity": "HIGH", "reason": "Warfarin + Metronidazole: CYP2C9 inhibition — major INR potentiation"},

    # Anticoagulant + fluoroquinolone (moderate INR effect)
    frozenset({"warfarin", "ciprofloxacin"}):  {"severity": "MODERATE", "reason": "Warfarin + Ciprofloxacin: gut flora disruption may elevate INR — monitor"},
    frozenset({"warfarin", "levofloxacin"}):   {"severity": "MODERATE", "reason": "Warfarin + Levofloxacin: monitor INR; fluoroquinolones can alter gut flora"},

    # QT prolongation pairs
    frozenset({"moxifloxacin", "amiodarone"}):  {"severity": "CRITICAL", "reason": "Moxifloxacin + Amiodarone: additive QT prolongation — risk of Torsades de Pointes"},
    frozenset({"levofloxacin", "amiodarone"}):  {"severity": "HIGH",     "reason": "Levofloxacin + Amiodarone: additive QT prolongation risk"},
    frozenset({"azithromycin", "amiodarone"}):  {"severity": "HIGH",     "reason": "Azithromycin + Amiodarone: additive QT prolongation"},

    # SSRI + NSAID (serotonin + antiplatelet — GI bleed)
    frozenset({"sertraline", "ibuprofen"}):   {"severity": "HIGH", "reason": "SSRI + NSAID: serotonin-mediated platelet inhibition + NSAID GI damage — bleeding risk"},
    frozenset({"fluoxetine", "ibuprofen"}):   {"severity": "HIGH", "reason": "SSRI + NSAID: major GI bleeding risk"},
    frozenset({"sertraline", "naproxen"}):    {"severity": "HIGH", "reason": "SSRI + NSAID: GI bleeding risk"},

    # Metformin + contrast/iodine (lactic acidosis risk)
    frozenset({"metformin", "iodinated-contrast"}): {"severity": "HIGH", "reason": "Metformin + iodinated contrast: hold metformin 48h — lactic acidosis risk (ADA guideline)"},

    # ACE inhibitor + potassium-sparing
    frozenset({"lisinopril", "spironolactone"}):  {"severity": "HIGH", "reason": "ACE inhibitor + spironolactone: hyperkalemia risk — monitor K+ closely"},
    frozenset({"enalapril", "spironolactone"}):   {"severity": "HIGH", "reason": "ACE inhibitor + spironolactone: hyperkalemia risk"},

    # Statin + macrolide (myopathy risk via CYP3A4)
    frozenset({"simvastatin", "clarithromycin"}):   {"severity": "HIGH",   "reason": "Simvastatin + Clarithromycin: CYP3A4 inhibition → statin myopathy/rhabdomyolysis risk"},
    frozenset({"simvastatin", "azithromycin"}):     {"severity": "MODERATE","reason": "Simvastatin + Azithromycin: moderate CYP3A4 inhibition — monitor for myopathy"},
    frozenset({"atorvastatin", "clarithromycin"}):  {"severity": "HIGH",   "reason": "Atorvastatin + Clarithromycin: CYP3A4 inhibition → myopathy risk"},
}


def apply_severity_overrides(interactions: list[dict]) -> list[dict]:
    """
    Upgrade interaction severity for known clinically dangerous drug pairs.
    Called after FDA label interaction detection, before LLM enrichment.
    Adds 'severity_upgraded' and 'upgrade_reason' fields for transparency.
    """
    upgraded = []
    for interaction in interactions:
        drug_a = interaction.get("drug_a", "").lower().split()[0].strip().rstrip(".,")
        drug_b = interaction.get("drug_b", "").lower().split()[0].strip().rstrip(".,")
        pair = frozenset({drug_a, drug_b})

        override = INTERACTION_SEVERITY_OVERRIDE.get(pair)
        if override:
            old_sev = interaction.get("severity", "MODERATE")
            new_sev = override["severity"]
            # Only upgrade — never downgrade
            sev_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
            if sev_rank.get(new_sev, 0) > sev_rank.get(old_sev, 0):
                interaction = {
                    **interaction,
                    "severity": new_sev,
                    "severity_upgraded": True,
                    "upgrade_reason": override["reason"],
                    "original_severity": old_sev,
                }
        upgraded.append(interaction)
    return upgraded


# ══════════════════════════════════════════════════════════════════════════════
# NEW v3.0: Critical lab assessment
# ══════════════════════════════════════════════════════════════════════════════

class CriticalLabFlag(BaseModel):
    """A single flagged lab value with clinical interpretation."""
    loinc: str
    display: str
    value: float
    unit: str
    flag: str                    # CRITICAL | HIGH | LOW
    clinical_meaning: str        # Plain-English interpretation
    drug_safety_implication: str # How this affects the drug check
    urgency: str                 # IMMEDIATE | URGENT | ROUTINE


class ClinicalLabContext(BaseModel):
    """Structured lab context injected into LLM prompts and FHIR notes."""
    critical_flags: list[CriticalLabFlag]
    sepsis_suspicion: bool
    renal_impairment_suspected: bool
    hepatic_impairment_suspected: bool
    coagulopathy_suspected: bool
    overall_lab_summary: str
    drug_selection_constraints: list[str]  # e.g. "avoid renally-cleared drugs"


# Reference ranges for common labs used in drug safety decisions
_LAB_RULES = [
    # (loinc_prefix_or_display_keyword, low_crit, low_concern, high_concern, high_crit, unit_hint)
    # WBC — sepsis marker
    {
        "keywords": ["wbc", "white blood cell", "leukocyte"],
        "high_critical": 20.0, "high_concern": 11.0,
        "low_concern": 3.5, "low_critical": 2.0,
        "unit": "10*3/uL",
        "high_crit_meaning": "Leukocytosis suggesting sepsis or severe infection — antibiotic urgency elevated",
        "high_concern_meaning": "Elevated WBC — active infection likely",
        "low_crit_meaning": "Severe leukopenia — immunocompromised, avoid myelosuppressive drugs",
        "drug_safety_implication_high": "Antibiotic selection is urgent; untreated sepsis increases drug toxicity risk",
        "drug_safety_implication_low": "Avoid drugs with myelosuppressive potential",
    },
    # Serum creatinine — renal function
    {
        "keywords": ["creatinine", "serum creatinine"],
        "high_critical": 4.0, "high_concern": 1.5,
        "low_concern": None, "low_critical": None,
        "unit": "mg/dL",
        "high_crit_meaning": "Severe renal impairment — renally-cleared drugs accumulate",
        "high_concern_meaning": "Reduced renal function — dose-adjust renally-cleared drugs",
        "drug_safety_implication_high": "Avoid NSAIDs, metformin (if eGFR <30), renally-cleared antibiotics without dose adjustment",
        "drug_safety_implication_low": None,
    },
    # Potassium
    {
        "keywords": ["potassium", "serum potassium", "k+"],
        "high_critical": 6.0, "high_concern": 5.5,
        "low_concern": 3.0, "low_critical": 2.5,
        "unit": "mEq/L",
        "high_crit_meaning": "Hyperkalemia — risk of arrhythmia, avoid K-sparing drugs",
        "high_concern_meaning": "Borderline hyperkalemia",
        "low_crit_meaning": "Severe hypokalemia — QT prolongation risk amplified by fluoroquinolones/macrolides",
        "drug_safety_implication_high": "Avoid ACE inhibitors, K-sparing diuretics, NSAIDs",
        "drug_safety_implication_low": "Hypokalemia amplifies QT prolongation risk — avoid fluoroquinolones/macrolides",
    },
    # INR / PT
    {
        "keywords": ["inr", "prothrombin", "pt "],
        "high_critical": 3.5, "high_concern": 3.0,
        "low_concern": None, "low_critical": None,
        "unit": "ratio",
        "high_crit_meaning": "Supratherapeutic anticoagulation — major bleeding risk",
        "high_concern_meaning": "Elevated INR — increased bleeding risk with anticoagulants",
        "drug_safety_implication_high": "Avoid all drugs that elevate INR (NSAIDs, macrolides, metronidazole)",
        "drug_safety_implication_low": None,
    },
    # ALT/AST — hepatic function
    {
        "keywords": ["alt", "alanine aminotransferase", "sgpt"],
        "high_critical": 500, "high_concern": 120,
        "low_concern": None, "low_critical": None,
        "unit": "U/L",
        "high_crit_meaning": "Severe hepatotoxicity — hepatically-metabolised drugs accumulate",
        "high_concern_meaning": "Elevated ALT — caution with hepatotoxic drugs",
        "drug_safety_implication_high": "Avoid clarithromycin, azithromycin, statins, high-dose acetaminophen",
        "drug_safety_implication_low": None,
    },
    {
        "keywords": ["ast", "aspartate aminotransferase", "sgot"],
        "high_critical": 500, "high_concern": 120,
        "low_concern": None, "low_critical": None,
        "unit": "U/L",
        "high_crit_meaning": "Severe hepatocellular injury",
        "high_concern_meaning": "Elevated AST — monitor hepatotoxic drugs",
        "drug_safety_implication_high": "Caution with hepatically-cleared drugs",
        "drug_safety_implication_low": None,
    },
    # Hemoglobin — bleeding context
    {
        "keywords": ["hemoglobin", "haemoglobin", "hgb", "hb"],
        "high_critical": None, "high_concern": None,
        "low_concern": 10.0, "low_critical": 7.0,
        "unit": "g/dL",
        "high_crit_meaning": None,
        "high_concern_meaning": None,
        "low_crit_meaning": "Severe anemia — any drug increasing bleeding risk is especially dangerous",
        "drug_safety_implication_high": None,
        "drug_safety_implication_low": "Avoid NSAIDs, antiplatelet agents — already anemic",
    },
    # Platelets
    {
        "keywords": ["platelet", "plt"],
        "high_critical": None, "high_concern": None,
        "low_concern": 100, "low_critical": 50,
        "unit": "10*3/uL",
        "high_crit_meaning": None,
        "high_concern_meaning": None,
        "low_crit_meaning": "Severe thrombocytopenia — avoid drugs that impair platelet function",
        "drug_safety_implication_high": None,
        "drug_safety_implication_low": "Avoid NSAIDs, aspirin — thrombocytopenia amplifies bleeding risk",
    },
    # eGFR
    {
        "keywords": ["egfr", "glomerular filtration"],
        "high_critical": None, "high_concern": None,
        "low_concern": 45, "low_critical": 30,
        "unit": "mL/min",
        "high_crit_meaning": None,
        "high_concern_meaning": None,
        "low_crit_meaning": "Severely reduced eGFR — renally-cleared drugs contraindicated",
        "drug_safety_implication_high": None,
        "drug_safety_implication_low": "Metformin contraindicated if eGFR <30; dose-adjust aminoglycosides, fluoroquinolones",
    },
]


def _match_lab_rule(lab: dict) -> Optional[dict]:
    """Match a lab result dict to a rule in _LAB_RULES by keyword matching display name."""
    display = (lab.get("display") or "").lower()
    loinc = (lab.get("loinc") or "").lower()
    for rule in _LAB_RULES:
        for kw in rule["keywords"]:
            if kw in display or kw in loinc:
                return rule
    return None


def assess_critical_labs(lab_results: list[dict]) -> ClinicalLabContext:
    """
    Assess patient lab results for values that affect drug safety decisions.
    Deterministic — no LLM. Runs before LLM enrichment so context is available.

    Returns a ClinicalLabContext with:
    - Structured flags for each abnormal value
    - Boolean markers (sepsis suspicion, renal impairment, etc.)
    - A list of drug selection constraints for the LLM prompts
    - A plain-English summary for FHIR notes
    """
    critical_flags: list[CriticalLabFlag] = []
    constraints: list[str] = []
    sepsis = False
    renal = False
    hepatic = False
    coagulopathy = False

    for lab in lab_results:
        rule = _match_lab_rule(lab)
        if not rule:
            continue

        value = lab.get("value")
        if value is None:
            continue

        flag = lab.get("flag", "")
        display = lab.get("display", rule["keywords"][0])
        unit = lab.get("unit", rule.get("unit", ""))
        loinc = lab.get("loinc", "")

        clinical_meaning = None
        drug_impl = None
        flag_level = None

        # Check high critical
        if rule["high_critical"] and value >= rule["high_critical"]:
            clinical_meaning = rule["high_crit_meaning"]
            drug_impl = rule["drug_safety_implication_high"]
            flag_level = "CRITICAL"
        # Check high concern
        elif rule["high_concern"] and value >= rule["high_concern"]:
            clinical_meaning = rule.get("high_concern_meaning", rule["high_crit_meaning"])
            drug_impl = rule["drug_safety_implication_high"]
            flag_level = "HIGH"
        # Check low critical
        elif rule["low_critical"] and value <= rule["low_critical"]:
            clinical_meaning = rule.get("low_crit_meaning")
            drug_impl = rule.get("drug_safety_implication_low")
            flag_level = "CRITICAL"
        # Check low concern
        elif rule["low_concern"] and value <= rule["low_concern"]:
            clinical_meaning = rule.get("low_concern_meaning", rule.get("low_crit_meaning"))
            drug_impl = rule.get("drug_safety_implication_low")
            flag_level = "HIGH"

        if flag_level and clinical_meaning:
            urgency = "IMMEDIATE" if flag_level == "CRITICAL" else "URGENT"
            critical_flags.append(CriticalLabFlag(
                loinc=loinc,
                display=display,
                value=value,
                unit=unit,
                flag=flag_level,
                clinical_meaning=clinical_meaning,
                drug_safety_implication=drug_impl or "Review drug selection in context of this abnormal value",
                urgency=urgency,
            ))
            if drug_impl:
                constraints.append(f"{display}={value} {unit}: {drug_impl}")

        # Set boolean markers
        keywords = rule["keywords"]
        if any(k in keywords for k in ["wbc", "white blood cell"]) and value >= 15.0:
            sepsis = True
        if any(k in keywords for k in ["creatinine", "egfr"]):
            if (rule["high_concern"] and value >= rule["high_concern"]) or \
               (rule["low_concern"] and value <= rule["low_concern"]):
                renal = True
        if any(k in keywords for k in ["alt", "ast"]) and value >= 120:
            hepatic = True
        if any(k in keywords for k in ["inr", "platelet"]):
            if flag_level in ("CRITICAL", "HIGH"):
                coagulopathy = True

    # Build plain-English summary
    if critical_flags:
        flag_strs = [f"{f.display} {f.value} {f.unit} ({f.flag})" for f in critical_flags]
        summary = f"Critical lab values: {'; '.join(flag_strs)}."
        if sepsis:
            summary += " Sepsis suspected — antibiotic urgency elevated."
        if renal:
            summary += " Renal impairment suspected — avoid nephrotoxic/renally-cleared drugs."
        if coagulopathy:
            summary += " Coagulopathy risk — avoid drugs that elevate bleeding."
    else:
        summary = "No critical lab values identified in provided results."

    return ClinicalLabContext(
        critical_flags=critical_flags,
        sepsis_suspicion=sepsis,
        renal_impairment_suspected=renal,
        hepatic_impairment_suspected=hepatic,
        coagulopathy_suspected=coagulopathy,
        overall_lab_summary=summary,
        drug_selection_constraints=constraints,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models for structured LLM output (unchanged from v2.0)
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
        description="Top 3 patient-specific factors driving the risk"
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
# Allergy Cross-Reactivity Table (unchanged from v2.0)
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

CROSS_REACTIVITY_SEVERITY: dict[str, dict[str, str]] = {
    "penicillin": {
        "penicillin":   "CRITICAL",
        "amoxicillin":  "CRITICAL",
        "cephalosporin":"MODERATE",
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
    "sulfa": {"sulfa": "CRITICAL"},
    "nsaid": {"nsaid": "HIGH"},
    "fluoroquinolone": {"fluoroquinolone": "HIGH"},
    "macrolide": {"macrolide": "MODERATE"},
    "tetracycline": {"tetracycline": "MODERATE"},
    "aminoglycoside": {"aminoglycoside": "HIGH"},
    "opioid": {"opioid": "MODERATE"},
    "benzodiazepine": {"benzodiazepine": "MODERATE"},
    "statin": {"statin": "LOW"},
}


def normalize_drug_name(name: str) -> str:
    return name.lower().split()[0].strip().rstrip(".,")


def get_drug_family(drug_name: str) -> Optional[str]:
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
    """Deterministic allergy cross-reactivity check. Unchanged from v2.0."""
    contraindications = []
    drug_lower = normalize_drug_name(drug_name)
    drug_family = get_drug_family(drug_name)

    for allergy in patient_allergies:
        allergen = allergy.get("substance", "").lower().strip()
        allergy_severity = allergy.get("severity", "unknown").lower()
        reaction = allergy.get("reaction", "unknown")

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

        allergen_family = get_drug_family(allergen) or allergen

        if drug_family and allergen_family:
            cross_sev = (
                CROSS_REACTIVITY_SEVERITY
                .get(allergen_family, {})
                .get(drug_family)
            )

            if cross_sev:
                if cross_sev == "MODERATE" and allergy_severity not in (
                    "severe", "high", "anaphylaxis", "anaphylactic"
                ):
                    continue
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
    """Deterministic condition contraindication check. Unchanged from v2.0."""
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


import logging
logger = logging.getLogger("safety_core")

# ══════════════════════════════════════════════════════════════════════════════
# LLM: Interaction enrichment (unchanged prompt, lab context added)
# ═══════════════════════════════════════════════════════════════════════════════

_INTERACTION_ENRICHMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior clinical pharmacist providing structured drug interaction analysis. "
     "Be precise, evidence-based, and clinically actionable. Do not include disclaimers."),
    ("human",
     """Patient context:
- Age: {age}, Gender: {gender}
- Active conditions: {conditions}
- Current medications: {current_medications}
- Critical lab context: {lab_context}

Drug interactions detected:
{interactions_json}

For each interaction listed above (in the same order), provide clinical enrichment.
Also write an overall risk narrative for this specific patient."""),
])


async def enrich_interactions_with_llm(
    interactions: list[dict],
    patient_state: dict,
    lab_context: Optional["ClinicalLabContext"] = None,
) -> Optional[InteractionEnrichmentBatch]:
    """LLM enrichment for interactions. Now accepts lab_context for richer prompts."""
    if not interactions:
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

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
        lab_str = lab_context.overall_lab_summary if lab_context else "No lab data provided"

        interactions_summary = [
            {
                "drug_a": i.get("drug_a"),
                "drug_b": i.get("drug_b"),
                "severity": i.get("severity"),
                "rxnav_description": i.get("description", ""),
                "severity_upgraded": i.get("severity_upgraded", False),
                "upgrade_reason": i.get("upgrade_reason", ""),
            }
            for i in interactions
        ]

        chain = _INTERACTION_ENRICHMENT_PROMPT | structured_llm
        result = await chain.ainvoke({
            "age": demographics.get("age", "unknown"),
            "gender": demographics.get("gender", "unknown"),
            "conditions": conditions_str,
            "current_medications": meds_str,
            "lab_context": lab_str,
            "interactions_json": json.dumps(interactions_summary, indent=2),
        })

        return result

    except Exception as e:
        logger.error(f" ✘   Interaction enrichment LLM failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM: Patient risk profile (lab context injected)
# ══════════════════════════════════════════════════════════════════════════════

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

Critical laboratory values:
{lab_context}

Drug safety findings:
- Proposed medications: {proposed_meds}
- Contraindications found ({contra_count}): {contraindications_json}
- Drug interactions found ({interaction_count}): {interactions_json}
- FDA black box warnings: {fda_warnings_json}
- Overall safety status: {safety_status}

Generate a patient-specific risk profile and clinical summary.
If critical lab values suggest sepsis or urgent treatment need, note this explicitly."""),
])


async def generate_patient_risk_profile(
    patient_state: dict,
    proposed_medications: list[str],
    contraindications: list[dict],
    interactions: list[dict],
    fda_warnings: dict,
    safety_status: str,
    lab_context: Optional["ClinicalLabContext"] = None,
) -> Optional[PatientRiskProfile]:
    """Generate patient risk profile. Now accepts lab_context."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

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

        renal_conditions = [
            c for c in patient_state.get("active_conditions", [])
            if c.get("code", "").startswith("N18") or c.get("code", "").startswith("N17")
        ]
        renal_hint = (
            f"Renal impairment noted: {renal_conditions[0].get('display')}"
            if renal_conditions else "No renal impairment conditions documented"
        )
        if lab_context and lab_context.renal_impairment_suspected:
            renal_hint += " (also suspected from lab values)"

        lab_str = lab_context.overall_lab_summary if lab_context else "No lab data provided"
        if lab_context and lab_context.drug_selection_constraints:
            lab_str += " Constraints: " + "; ".join(lab_context.drug_selection_constraints)

        result = await chain.ainvoke({
            "age": demographics.get("age", "unknown"),
            "gender": demographics.get("gender", "unknown"),
            "conditions": conditions_str,
            "current_medications": meds_str,
            "allergies": allergies_str,
            "renal_hint": renal_hint,
            "lab_context": lab_str,
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
                {k: v[:2] for k, v in list(fda_warnings.items())[:3] if v}
            ),
            "safety_status": safety_status,
        })

        return result

    except Exception as e:
        logger.error(f" ✘   Risk profile LLM failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NEW v3.0: Proactive alternatives for ALL flagged drugs
# ══════════════════════════════════════════════════════════════════════════════

_PROACTIVE_ALTERNATIVES_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior clinical pharmacist recommending medication alternatives. "
     "Suggestions must be specific, avoid the patient's known allergens, "
     "account for critical lab values, and be immediately actionable."),
    ("human",
     """A patient cannot take {drug_name} because: {reason_for_avoidance}

Drug indication: {indication}

Patient context:
- Age: {age}, Gender: {gender}
- Active ICD-10 conditions: {patient_conditions}
- Current medications: {current_medications}
- Known allergies: {allergies}
- Critical lab context: {lab_context}
- Sepsis suspected: {sepsis_flag}

Additional safety constraints from lab values:
{lab_constraints}

Suggest 2-3 safe alternatives ranked by clinical preference.
- Avoid drugs in the same allergy family as {allergen_family}
- Consider renal/hepatic function for dose recommendations
- If sepsis is suspected, prioritise broad-spectrum coverage
- Include any specific monitoring needed for each alternative"""),
])


async def generate_proactive_alternatives(
    drug_name: str,
    contraindications_for_drug: list[dict],
    patient_state: dict,
    lab_context: Optional[ClinicalLabContext] = None,
    active_conditions: Optional[list[dict]] = None,
    current_medications: Optional[list[dict]] = None,
    allergies: Optional[list[dict]] = None,
) -> Optional[AlternativeSuggestionsOutput]:
    """
    NEW v3.0: Generate proactive alternatives for a flagged drug.
    Called for every flagged drug at end of pipeline — not just on request.
    Accepts lab context to steer away from drugs that are unsafe given lab values.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

    try:
        structured_llm = llm.with_structured_output(AlternativeSuggestionsOutput)
        chain = _PROACTIVE_ALTERNATIVES_PROMPT | structured_llm

        # Derive reason and indication from contraindications
        reasons = [c.get("reason", "") for c in contraindications_for_drug]
        reason_str = "; ".join(reasons[:3]) if reasons else "Safety concern identified"

        # Infer indication from conditions (best effort)
        conditions = active_conditions or patient_state.get("active_conditions", [])
        indication_parts = [c.get("display", c.get("code", "")) for c in conditions[:3]]
        indication = ", ".join(indication_parts) or "clinical indication not specified"

        # Allergen family for the primary allergen
        allergen_family = "unknown"
        for c in contraindications_for_drug:
            if c.get("type") in ("direct_allergy", "cross_reactivity") and c.get("allergen"):
                allergen_family = get_drug_family(c["allergen"]) or c["allergen"]
                break

        demographics = patient_state.get("demographics", {})
        conditions_str = ", ".join(
            f"{c.get('code')} ({c.get('display', '')})"
            for c in conditions[:5]
        ) or "None"
        meds = current_medications or patient_state.get("medications", [])
        meds_str = ", ".join(m.get("drug", "") for m in meds[:5]) or "None"
        allergy_list = allergies or patient_state.get("allergies", [])
        allergies_str = ", ".join(
            f"{a.get('substance')} (reaction: {a.get('reaction', '?')}, severity: {a.get('severity', '?')})"
            for a in allergy_list
        ) or "NKDA"

        lab_str = lab_context.overall_lab_summary if lab_context else "No lab data provided"
        lab_constraints = (
            "\n".join(f"- {c}" for c in lab_context.drug_selection_constraints)
            if lab_context and lab_context.drug_selection_constraints
            else "None"
        )
        sepsis_flag = "YES — prioritise broad-spectrum antibiotic coverage" if (
            lab_context and lab_context.sepsis_suspicion
        ) else "No"

        result = await chain.ainvoke({
            "drug_name": drug_name,
            "reason_for_avoidance": reason_str,
            "indication": indication,
            "allergen_family": allergen_family,
            "age": demographics.get("age", "unknown"),
            "gender": demographics.get("gender", "unknown"),
            "patient_conditions": conditions_str,
            "current_medications": meds_str,
            "allergies": allergies_str,
            "lab_context": lab_str,
            "lab_constraints": lab_constraints,
            "sepsis_flag": sepsis_flag,
        })

        return result

    except Exception as e:
        logger.error(f"  ✘  Proactive alternatives LLM failed for {drug_name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM: Alternative suggestions (MCP tool path — unchanged API)
# ══════════════════════════════════════════════════════════════════════════════

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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

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
        logger.error(f"  ✘  LLM alternatives failed: {e}")
        return None


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


# ══════════════════════════════════════════════════════════════════════════════
# UPDATED v3.0: FHIR MedicationRequest — drug-specific notes
# ══════════════════════════════════════════════════════════════════════════════

def build_fhir_medication_request(
    drug_name: str,
    patient_id: str,
    safety_cleared: bool,
    safety_note: str = "",
    risk_level: str = "LOW",
    specific_reason: str = "",       # NEW: per-drug contraindication reason
    lab_context_note: str = "",      # NEW: relevant lab findings for this drug
) -> dict:
    """
    Build a FHIR R4 MedicationRequest resource.

    v3.0 changes:
    - specific_reason: the exact contraindication reason for this drug (not generic text)
    - lab_context_note: any critical lab values relevant to this drug's safety
    """
    status = "active" if safety_cleared else "on-hold"
    priority = "urgent" if risk_level in ("CRITICAL", "HIGH") else "routine"

    # Build a drug-specific note rather than a generic one
    if safety_cleared:
        note_parts = [f"MediTwin Drug Safety Check (v3.0): CLEARED."]
        if safety_note:
            note_parts.append(safety_note)
        if lab_context_note:
            note_parts.append(f"Lab context: {lab_context_note}")
    else:
        note_parts = [f"MediTwin Drug Safety Check (v3.0): FLAGGED — DO NOT DISPENSE."]
        if specific_reason:
            note_parts.append(f"Reason: {specific_reason}")
        elif safety_note:
            note_parts.append(safety_note)
        if lab_context_note:
            note_parts.append(f"Lab context: {lab_context_note}")

    note_parts.append(" ⚠  AI-generated — requires physician verification before dispensing.")

    return {
        "resourceType": "MedicationRequest",
        "status": status,
        "intent": "proposal",
        "priority": priority,
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {"text": drug_name},
        "extension": [
            {
                "url": "https://meditwin.ai/fhir/StructureDefinition/ai-safety-check",
                "valueString": "CLEARED" if safety_cleared else "FLAGGED"
            }
        ],
        "note": [{"text": " ".join(note_parts)}],
    }