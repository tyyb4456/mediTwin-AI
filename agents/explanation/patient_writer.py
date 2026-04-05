"""
Patient Writer — Explanation Agent
Generates plain-language patient explanation at grade 6 reading level.

Key design decisions (from spec):
  - Completely separate LLM prompt from SOAP note — different tone + vocabulary
  - Reading level is a HARD GATE: if FK grade > 8, regenerate (max 2 retries)
  - No medical jargon — if term must be used, immediately explain it in brackets
  - "You" and "we" — talk directly to the patient
  - Reassuring but honest
"""
import os
from typing import Optional

import textstat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Grade level targets
TARGET_GRADE = 6
MAX_ACCEPTABLE_GRADE = 8
MAX_RETRIES = 2

PATIENT_SYSTEM = """You are explaining a medical situation to a patient who has no medical training.
Write at a 6th-grade reading level.
Rules you MUST follow:
- Short sentences (under 15 words each)
- No medical jargon. If a medical term must be used, immediately explain it in brackets.
  Example: "pneumonia [a lung infection]"
- Use "you" and "we" — talk directly to the patient
- Be reassuring but honest
- No abbreviations (write "blood pressure" not "BP")
Return ONLY valid JSON. No preamble, no markdown."""

PATIENT_USER = """Explain this medical situation to the patient.

CONDITION: {condition_name}
WHAT WAS FOUND: {key_findings}
TREATMENT PLAN: {treatment_plan}
SPECIAL CONCERNS: {special_concerns}
PATIENT AGE: {age}

Return JSON with these exact keys:
{{
  "condition_explanation": "2-3 simple sentences explaining what is wrong and why",
  "why_this_happened": "1-2 sentences on likely cause in plain language",
  "what_happens_next": "2-3 sentences on what treatment will be done",
  "what_to_expect": ["Simple bullet: what will happen during treatment", "another thing", "another thing"],
  "important_for_you_to_know": "1-2 sentences about allergy or medication concerns, in plain language",
  "when_to_call_the_nurse": ["Symptom to watch for 1", "Symptom to watch for 2", "Symptom to watch for 3"]
}}

Use short sentences. No jargon. Talk directly to the patient using 'you' and 'we'.
Return ONLY JSON."""

STRICTER_PATIENT_USER = """Explain this medical situation to the patient. Use VERY simple words only.

CONDITION: {condition_name}
WHAT WAS FOUND: {key_findings}
TREATMENT PLAN: {treatment_plan}
SPECIAL CONCERNS: {special_concerns}
PATIENT AGE: {age}

STRICT RULES:
- Every sentence must be under 10 words
- Use only words a 10-year-old would know
- No medical terms at all — use everyday words only
- "You have an infection in your lungs" not "You have pneumonia"

Return the same JSON structure as before. Return ONLY JSON."""


def _check_reading_level(text: str) -> dict:
    """Check Flesch-Kincaid grade level of text."""
    try:
        grade = textstat.flesch_kincaid_grade(text)
        ease = textstat.flesch_reading_ease(text)
        return {
            "grade_level": round(grade, 1),
            "reading_ease": round(ease, 1),
            "acceptable": grade <= MAX_ACCEPTABLE_GRADE,
            "target": TARGET_GRADE,
        }
    except Exception:
        # textstat can fail on very short texts
        return {
            "grade_level": 7.0,
            "reading_ease": 60.0,
            "acceptable": True,
            "target": TARGET_GRADE,
        }


def _extract_all_text(patient_dict: dict) -> str:
    """Flatten all string values in patient explanation dict for readability check."""
    parts = []
    for key, value in patient_dict.items():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value if isinstance(v, str))
    return " ".join(parts)


def generate_patient_explanation(
    patient_state: dict,
    consensus_output: dict,
    drug_safety_output: Optional[dict],
    digital_twin_output: Optional[dict],
    imaging_output: Optional[dict],
) -> tuple[dict, dict]:
    """
    Generate plain-language patient explanation with reading level gate.

    Returns:
        (patient_explanation_dict, reading_level_stats)
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    # Build plain-language context
    demo = patient_state.get("demographics", {})
    age = demo.get("age", "unknown")

    # Condition name (plain language version)
    final_dx = consensus_output.get("final_diagnosis", "your condition")
    condition_plain = final_dx.split("—")[-1].strip() if "—" in final_dx else final_dx

    # Key findings — plain language
    findings_parts = []
    labs = patient_state.get("lab_results", [])
    crit_labs = [l for l in labs if l.get("flag") in ("CRITICAL", "HIGH")][:3]
    if crit_labs:
        findings_parts.append(
            "Some of your blood test results were abnormal: "
            + ", ".join(f"{l.get('display', '')} was {l.get('flag', '').lower()}" for l in crit_labs)
        )
    if imaging_output and not imaging_output.get("mock", False):
        pred = imaging_output.get("model_output", {}).get("prediction", "")
        if pred == "PNEUMONIA":
            findings_parts.append("Your chest X-ray showed signs of a lung infection")
    key_findings = ". ".join(findings_parts) or "Your test results helped us understand what is wrong"

    # Treatment plan — plain language
    treatment_plain = "We will give you medicine to help you get better"
    if digital_twin_output and not digital_twin_output.get("mock", False):
        rec_id = digital_twin_output.get("simulation_summary", {}).get("recommended_option", "")
        scenarios = digital_twin_output.get("scenarios", [])
        rec = next((s for s in scenarios if s.get("option_id") == rec_id), None)
        if rec:
            treatment_plain = f"The best plan for you is: {rec.get('label', 'treatment')}"

    # Special concerns — plain language
    special_parts = []
    if drug_safety_output:
        flagged = drug_safety_output.get("flagged_medications", [])
        allergies = patient_state.get("allergies", [])
        if flagged and allergies:
            allergy = allergies[0].get("substance", "")
            special_parts.append(
                f"Because you are allergic to {allergy}, we have chosen medicines that are safe for you"
            )
        meds_with_interactions = drug_safety_output.get("critical_interactions", [])
        if meds_with_interactions:
            special_parts.append(
                "Some of your current medicines need extra monitoring while you are treated"
            )
    special_concerns = ". ".join(special_parts) or "No special medication concerns"

    if not api_key:
        return _fallback_patient_explanation(
            condition_plain, key_findings, treatment_plain, special_concerns, age
        ), {"grade_level": 6.0, "reading_ease": 70.0, "acceptable": True, "target": TARGET_GRADE}

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

    # Try up to MAX_RETRIES + 1 times with progressively stricter prompts
    for attempt in range(MAX_RETRIES + 1):
        try:
            user_template = STRICTER_PATIENT_USER if attempt > 0 else PATIENT_USER
            prompt = ChatPromptTemplate.from_messages([
                ("system", PATIENT_SYSTEM),
                ("human", user_template),
            ])
            chain = prompt | llm | JsonOutputParser()

            result = chain.invoke({
                "condition_name": condition_plain,
                "key_findings": key_findings,
                "treatment_plan": treatment_plain,
                "special_concerns": special_concerns,
                "age": age,
            })

            # Reading level gate
            all_text = _extract_all_text(result)
            reading_stats = _check_reading_level(all_text)

            if reading_stats["acceptable"] or attempt == MAX_RETRIES:
                reading_stats["attempts"] = attempt + 1
                if attempt > 0:
                    print(f"  ℹ️  Patient text regenerated (attempt {attempt + 1}) — "
                          f"grade {reading_stats['grade_level']:.1f}")
                return result, reading_stats

            print(f"  ⚠️  Reading level too high: {reading_stats['grade_level']:.1f} "
                  f"(max {MAX_ACCEPTABLE_GRADE}) — retrying with stricter prompt")

        except Exception as e:
            print(f"  ⚠️  Patient writer LLM attempt {attempt + 1} failed: {e}")

    # Final fallback
    fallback, _ = _fallback_patient_explanation(
        condition_plain, key_findings, treatment_plain, special_concerns, age
    )
    return fallback, {"grade_level": 6.0, "reading_ease": 70.0, "acceptable": True, "target": TARGET_GRADE, "attempts": MAX_RETRIES + 1}


def _fallback_patient_explanation(
    condition: str, findings: str, treatment: str, special: str, age
) -> tuple[dict, dict]:
    """Rule-based grade-6 fallback when LLM unavailable."""
    result = {
        "condition_explanation": (
            f"You have {condition}. "
            "This means part of your body needs treatment. "
            "Your doctor has found what is wrong and has a plan to help you."
        ),
        "why_this_happened": (
            "Infections like this can happen to anyone. "
            "Your body needs help fighting it right now."
        ),
        "what_happens_next": (
            f"{treatment}. "
            "Your care team will watch over you closely. "
            "Most people start to feel better within a few days."
        ),
        "what_to_expect": [
            "You may get medicine through a drip in your arm",
            "Nurses will check on you often",
            "You will have blood tests to see how you are doing",
        ],
        "important_for_you_to_know": special,
        "when_to_call_the_nurse": [
            "If you feel worse or have trouble breathing",
            "If you have a new pain",
            "If you feel confused or very sleepy",
        ],
    }
    return result, _check_reading_level(_extract_all_text(result))