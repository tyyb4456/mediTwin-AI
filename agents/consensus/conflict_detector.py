"""
Conflict Detector — Consensus Agent
Pure Python rules engine. Deterministic. No LLM.
Detects three conflict types as specified in the architecture doc.

Conflict types:
  1. diagnosis_lab_mismatch     — Diagnosis Agent and Lab Agent disagree on top diagnosis
  2. imaging_clinical_dissociation — Imaging says normal but clinical suspicion is high
  3. treatment_contraindicated  — Drug Safety Agent rejects the proposed treatment

Severity levels: LOW | MODERATE | HIGH
  HIGH → escalate
  MODERATE → attempt tiebreaker RAG
  LOW → note in output, still reach consensus
"""
from dataclasses import dataclass, field
from typing import Optional


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Conflict:
    type: str              # conflict type key
    severity: str          # LOW | MODERATE | HIGH
    description: str
    agent_a: str = ""
    output_a: str = ""
    agent_b: str = ""
    output_b: str = ""
    confidence_a: float = 0.0
    confidence_b: float = 0.0
    extra: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _icd_prefix(code: str) -> str:
    """Return the 3-char ICD-10 category prefix. 'J18.9' → 'J18'"""
    return code[:3] if code else ""


def _codes_compatible(code_a: str, code_b: str) -> bool:
    """
    Two ICD-10 codes are compatible if they share the same 3-char category prefix.
    J18.9 and J18.1 → compatible (both pneumonia).
    J18.9 and J12.9 → NOT compatible (bacterial vs viral).
    """
    if not code_a or not code_b:
        return True   # Can't compare what we don't have — don't flag
    return _icd_prefix(code_a) == _icd_prefix(code_b)


def _compute_conflict_severity(conf_a: float, conf_b: float) -> str:
    """
    Severity of a diagnosis conflict based on both agents' confidence.
    Both confident + disagreeing = HIGH.
    One uncertain = MODERATE.
    Both uncertain = LOW.
    """
    high_conf = 0.70
    if conf_a >= high_conf and conf_b >= high_conf:
        return "HIGH"
    elif conf_a >= high_conf or conf_b >= high_conf:
        return "MODERATE"
    else:
        return "LOW"


# ── Main conflict detection function ──────────────────────────────────────────

def detect_conflicts(
    diagnosis_output: Optional[dict],
    lab_output: Optional[dict],
    imaging_output: Optional[dict],
    drug_safety_output: Optional[dict],
) -> list[Conflict]:
    """
    Run all three conflict detection rules against agent outputs.
    Returns a list of Conflict objects (empty = no conflicts detected).

    Args are the raw JSON outputs from each specialist agent.
    Any can be None if that agent didn't run (e.g. no imaging).
    """
    conflicts: list[Conflict] = []

    # ── Rule 1: Diagnosis ↔ Lab disagreement ─────────────────────────────────
    if diagnosis_output and lab_output:
        dx_code = diagnosis_output.get("top_icd10_code", "")
        dx_conf = diagnosis_output.get("differential_diagnosis", [{}])[0].get("confidence", 0.0) \
                  if diagnosis_output.get("differential_diagnosis") else 0.0
        dx_display = diagnosis_output.get("top_diagnosis", "Unknown")

        lab_confirms = lab_output.get("diagnosis_confirmation", {}).get("confirms_top_diagnosis", True)
        lab_alt_code = lab_output.get("diagnosis_confirmation", {}).get("alternative_diagnosis_code")
        lab_boost = lab_output.get("diagnosis_confirmation", {}).get("lab_confidence_boost", 0.0)

        # Conflict: lab explicitly does NOT confirm AND suggests a different ICD-10 category
        if not lab_confirms and lab_alt_code and not _codes_compatible(dx_code, lab_alt_code):
            severity = _compute_conflict_severity(dx_conf, abs(lab_boost) + 0.5)
            conflicts.append(Conflict(
                type="diagnosis_lab_mismatch",
                severity=severity,
                description=(
                    f"Diagnosis Agent suggests {dx_display} ({dx_code}) "
                    f"but Lab Analysis suggests {lab_alt_code}. "
                    f"Lab pattern does not support the top diagnosis."
                ),
                agent_a="diagnosis",
                output_a=dx_code,
                agent_b="lab",
                output_b=lab_alt_code,
                confidence_a=dx_conf,
                confidence_b=abs(lab_boost),
                extra={"lab_reasoning": lab_output.get("diagnosis_confirmation", {}).get("reasoning", "")},
            ))

    # ── Rule 2: Imaging ↔ Clinical dissociation ───────────────────────────────
    if imaging_output and diagnosis_output:
        img_prediction = imaging_output.get("model_output", {}).get("prediction", "")
        img_confidence = imaging_output.get("model_output", {}).get("confidence", 0.0)
        img_mock = imaging_output.get("mock", False)

        dx_conf = diagnosis_output.get("differential_diagnosis", [{}])[0].get("confidence", 0.0) \
                  if diagnosis_output.get("differential_diagnosis") else 0.0
        dx_code = diagnosis_output.get("top_icd10_code", "")

        # Only flag if: imaging says NORMAL with decent confidence,
        # clinical suspicion is high, AND imaging is not mock
        # AND the diagnosis is respiratory (where imaging is expected to confirm)
        is_respiratory_dx = dx_code.startswith(("J", "R0"))
        imaging_normal_confident = img_prediction == "NORMAL" and img_confidence >= 0.70
        clinical_suspicion_high = dx_conf >= 0.65 and is_respiratory_dx

        if not img_mock and imaging_normal_confident and clinical_suspicion_high:
            conflicts.append(Conflict(
                type="imaging_clinical_dissociation",
                severity="HIGH",
                description=(
                    f"Chest X-ray appears NORMAL (confidence {img_confidence:.0%}) "
                    f"but clinical assessment strongly suggests {diagnosis_output.get('top_diagnosis', '')} "
                    f"(confidence {dx_conf:.0%}). Possible early pneumonia not yet visible on X-ray, "
                    f"or alternative diagnosis (PE, lung mass) must be excluded."
                ),
                agent_a="imaging",
                output_a="NORMAL",
                agent_b="diagnosis",
                output_b=dx_code,
                confidence_a=img_confidence,
                confidence_b=dx_conf,
                extra={
                    "recommended_actions": [
                        "Radiologist review of chest X-ray",
                        "Consider CT chest if clinical suspicion remains high",
                        "Consider pulmonary embolism workup (D-dimer, CT-PA)",
                    ]
                },
            ))

    # ── Rule 3: Drug Safety rejection ────────────────────────────────────────
    if drug_safety_output:
        safety_status = drug_safety_output.get("safety_status", "SAFE")
        contraindications = drug_safety_output.get("contraindications", [])
        interactions = drug_safety_output.get("critical_interactions", [])
        flagged = drug_safety_output.get("flagged_medications", [])

        # Only flag as a conflict if there are HIGH/CRITICAL severity issues
        critical_contras = [
            c for c in contraindications
            if c.get("severity") in ("CRITICAL", "HIGH")
        ]
        severe_interactions = [
            i for i in interactions
            if i.get("severity", "").upper() in ("HIGH", "CRITICAL")
        ]

        if safety_status == "UNSAFE" and (critical_contras or severe_interactions):
            severity = "HIGH" if critical_contras else "MODERATE"
            descriptions = []
            for c in critical_contras[:2]:
                descriptions.append(
                    f"{c.get('drug', '?')} contraindicated: {c.get('reason', '')[:80]}"
                )
            for i in severe_interactions[:2]:
                descriptions.append(
                    f"{i.get('drug_a', '?')} + {i.get('drug_b', '?')}: {i.get('severity', '')} interaction"
                )

            conflicts.append(Conflict(
                type="treatment_contraindicated",
                severity=severity,
                description=(
                    f"Drug Safety Agent flagged {len(flagged)} medication(s) as unsafe. "
                    + " | ".join(descriptions)
                ),
                agent_a="drug_safety",
                output_a="UNSAFE",
                agent_b="treatment_plan",
                output_b=", ".join(flagged[:3]),
                extra={
                    "flagged_medications": flagged,
                    "alternatives_available": len(drug_safety_output.get("alternatives", [])) > 0,
                    "approved_medications": drug_safety_output.get("approved_medications", []),
                },
            ))

    return conflicts


def get_max_severity(conflicts: list[Conflict]) -> Optional[str]:
    """Return the highest severity across all conflicts."""
    if not conflicts:
        return None
    order = {"HIGH": 2, "MODERATE": 1, "LOW": 0}
    return max(conflicts, key=lambda c: order.get(c.severity, 0)).severity


def route_consensus(conflicts: list[Conflict]) -> str:
    """
    Determine routing based on conflicts.

    Returns:
        "no_conflict"  → proceed directly to explanation
        "resolve"      → run tiebreaker RAG
        "escalate"     → flag for human review
    """
    if not conflicts:
        return "no_conflict"

    max_sev = get_max_severity(conflicts)

    if max_sev == "HIGH":
        return "escalate"
    else:
        # MODERATE or LOW — try to resolve
        # But if imaging-clinical dissociation is present, always escalate
        # (we can't resolve contradictory physical evidence with RAG alone)
        imaging_conflicts = [c for c in conflicts if c.type == "imaging_clinical_dissociation"]
        if imaging_conflicts:
            return "escalate"
        return "resolve"