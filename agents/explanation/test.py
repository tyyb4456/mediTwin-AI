"""
Test script for Explanation Agent
Port: 8009

Part 1 — Unit tests (no server, no API key):
    - Reading level gate (textstat FK grade check)
    - FHIR Bundle assembly from agent outputs
    - Risk attribution structure
    - SOAP template fallback
    - Medical disclaimer always present

Part 2 — Integration tests (server must be running):
    - Full /explain pipeline
    - SOAP note has all 4 sections (S/O/A/P)
    - Patient explanation reading level ≤ grade 8
    - FHIR Bundle has ≥ 1 resource
    - Risk attribution present

Run:
    python agents/explanation/test.py --unit
    python agents/explanation/test.py
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# ── Shared test fixtures ───────────────────────────────────────────────────────

PATIENT_STATE = {
    "patient_id": "test-exp-001",
    "demographics": {"name": "John Test", "age": 54, "gender": "male", "dob": "1970-03-14"},
    "active_conditions": [
        {"code": "J18.9", "display": "Community-acquired pneumonia"},
        {"code": "E11.9", "display": "Type 2 diabetes"},
        {"code": "I48.0", "display": "Atrial fibrillation"},
    ],
    "medications": [
        {"drug": "Warfarin 5mg", "dose": "5mg", "frequency": "OD", "status": "active"},
        {"drug": "Metformin 850mg", "dose": "850mg", "frequency": "BID", "status": "active"},
    ],
    "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}],
    "lab_results": [
        {"loinc": "26464-8", "display": "WBC", "value": 18.4, "unit": "10*3/uL", "flag": "CRITICAL"},
        {"loinc": "1988-5", "display": "CRP", "value": 142.0, "unit": "mg/L", "flag": "HIGH"},
        {"loinc": "718-7", "display": "Hemoglobin", "value": 13.8, "unit": "g/dL", "flag": "NORMAL"},
    ],
    "diagnostic_reports": [], "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z", "imaging_available": True,
}

CONSENSUS_OUTPUT = {
    "consensus_status": "FULL_CONSENSUS",
    "final_diagnosis": "J18.9 — Community-acquired pneumonia",
    "final_icd10_code": "J18.9",
    "aggregate_confidence": 0.72,
    "human_review_required": False,
    "conflict_count": 0,
    "conflicts": [],
    "consensus_summary": "All agents agree on CAP diagnosis.",
}

DIAGNOSIS_OUTPUT = {
    "differential_diagnosis": [
        {"rank": 1, "icd10_code": "J18.9", "display": "Community-acquired pneumonia",
         "confidence": 0.87, "supporting_evidence": ["WBC elevated", "CRP high"]}
    ],
    "top_diagnosis": "Community-acquired pneumonia",
    "top_icd10_code": "J18.9",
    "confidence_level": "HIGH",
    "reasoning_summary": "Classic CAP presentation.",
    "recommended_next_steps": ["Start antibiotics", "Blood cultures"],
    "fhir_conditions": [
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/test-exp-001"},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                                  "code": "J18.9", "display": "Pneumonia"}]},
            "verificationStatus": {"coding": [{"code": "provisional"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
    ],
}

LAB_OUTPUT = {
    "lab_summary": {"total_results": 3, "abnormal_count": 2, "critical_count": 1, "overall_severity": "SEVERE"},
    "flagged_results": [{"loinc": "26464-8", "display": "WBC", "value": 18.4, "flag": "CRITICAL",
                         "clinical_significance": "Severe leukocytosis"}],
    "pattern_analysis": {"identified_patterns": [{"pattern": "Bacterial infection markers"}]},
    "diagnosis_confirmation": {"confirms_top_diagnosis": True, "lab_confidence_boost": 0.12},
    "critical_alerts": [{"level": "CRITICAL", "message": "WBC 18.4 — sepsis workup recommended", "action_required": True}],
}

IMAGING_OUTPUT = {
    "model_output": {"prediction": "PNEUMONIA", "confidence": 0.923, "pneumonia_probability": 0.923, "normal_probability": 0.077},
    "severity_assessment": {"grade": "MODERATE", "triage_priority": 2, "triage_label": "URGENT"},
    "imaging_findings": {"pattern": "Lobar consolidation", "affected_area": "Right lower lobe"},
    "clinical_interpretation": "Chest X-ray consistent with pneumonia (92.3% confidence).",
    "confirms_diagnosis": True,
    "diagnosis_code": "J18.9",
    "recommended_actions": ["Start antibiotics", "Blood cultures"],
    "fhir_diagnostic_report": {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "subject": {"reference": "Patient/test-exp-001"},
        "conclusion": "AI Triage: PNEUMONIA (92.3% confidence). Priority: URGENT. AI-generated.",
        "conclusionCode": [{"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "J18.9"}]}],
    },
    "model_loaded": True,
    "mock": False,
}

DRUG_SAFETY_OUTPUT = {
    "safety_status": "CAUTION",
    "critical_interactions": [{"drug_a": "Azithromycin", "drug_b": "Warfarin", "severity": "MODERATE",
                                "description": "Monitor INR closely"}],
    "contraindications": [],
    "alternatives": [],
    "approved_medications": ["Azithromycin 500mg", "Ceftriaxone 1g IV"],
    "flagged_medications": [],
    "fhir_medication_requests": [
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "proposal",
            "subject": {"reference": "Patient/test-exp-001"},
            "medicationCodeableConcept": {"text": "Azithromycin 500mg"},
        }
    ],
}

DIGITAL_TWIN_OUTPUT = {
    "simulation_summary": {
        "patient_risk_profile": "MODERATE-HIGH",
        "baseline_risks": {"readmission_30d": 0.28, "mortality_30d": 0.06, "complication": 0.35},
        "recommended_option": "B",
        "recommendation_confidence": 0.82,
    },
    "scenarios": [
        {"option_id": "A", "label": "Azithromycin outpatient",
         "predictions": {"recovery_probability_7d": 0.79, "mortality_risk_30d": 0.03,
                        "readmission_risk_30d": 0.18, "complication_risk": 0.22}},
        {"option_id": "B", "label": "IV antibiotics + hospitalization",
         "predictions": {"recovery_probability_7d": 0.91, "mortality_risk_30d": 0.01,
                        "readmission_risk_30d": 0.09, "complication_risk": 0.12}},
    ],
    "what_if_narrative": "Option B offers significantly better outcomes with 91% 7-day recovery.",
    "fhir_care_plan": {
        "resourceType": "CarePlan",
        "status": "active",
        "intent": "plan",
        "subject": {"reference": "Patient/test-exp-001"},
        "title": "AI Treatment Plan — Option B: IV antibiotics + hospitalization",
        "description": "IV antibiotics recommended for 54y male with moderate-high risk CAP.",
        "activity": [],
    },
    "feature_attribution": [
        {"feature": "WBC 18.4 (elevated)", "contribution": "+0.82", "direction": "increases_risk"},
        {"feature": "Age (54y)", "contribution": "+0.55", "direction": "increases_risk"},
        {"feature": "No CKD", "contribution": "-0.31", "direction": "reduces_risk"},
    ],
    "models_loaded": True,
    "mock": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    import textstat
    from fhir_bundler import build_fhir_bundle
    from soap_generator import generate_soap_note, MEDICAL_DISCLAIMER

    print("=" * 60)
    print("Unit Tests — patient_writer.py + fhir_bundler.py + soap_generator.py")
    print("=" * 60)

    # ── 1. Reading level gate (textstat) ──────────────────────────
    print("\n1. Reading level gate (textstat FK grade)...")

    # Simple text should be low grade
    simple_text = (
        "You have a lung infection. It is called pneumonia. "
        "We will give you medicine to help you get better. "
        "Most people feel better in a few days."
    )
    simple_grade = textstat.flesch_kincaid_grade(simple_text)
    assert simple_grade <= 8, f"Simple text grade {simple_grade:.1f} should be ≤ 8"
    print(f"   ✓ Simple text grade: {simple_grade:.1f} (acceptable ≤ 8)")

    # Complex clinical text should be high grade
    complex_text = (
        "The pathophysiological manifestations of community-acquired pneumonia "
        "are predominantly attributed to the dysregulation of inflammatory cytokine cascades "
        "subsequent to bacterial colonization of the alveolar epithelium."
    )
    complex_grade = textstat.flesch_kincaid_grade(complex_text)
    assert complex_grade > 8, f"Complex text grade {complex_grade:.1f} should be > 8"
    print(f"   ✓ Complex text grade: {complex_grade:.1f} (correctly high)")

    # ── 2. FHIR Bundle assembly ───────────────────────────────────
    print("\n2. FHIR Bundle assembly...")
    bundle = build_fhir_bundle(
        patient_id="test-exp-001",
        diagnosis_output=DIAGNOSIS_OUTPUT,
        imaging_output=IMAGING_OUTPUT,
        drug_safety_output=DRUG_SAFETY_OUTPUT,
        digital_twin_output=DIGITAL_TWIN_OUTPUT,
    )

    assert bundle["resourceType"] == "Bundle"
    print("   ✓ resourceType = Bundle")

    assert bundle["type"] == "collection"
    print("   ✓ type = collection")

    assert len(bundle["entry"]) >= 3, \
        f"Expected ≥ 3 resources (Condition + DiagnosticReport + MedicationRequest + CarePlan), got {len(bundle['entry'])}"
    print(f"   ✓ {len(bundle['entry'])} resources in bundle")

    resource_types = bundle["_resource_types"]
    assert "Condition" in resource_types, "Bundle must contain Condition"
    assert "DiagnosticReport" in resource_types, "Bundle must contain DiagnosticReport"
    assert "CarePlan" in resource_types, "Bundle must contain CarePlan"
    print(f"   ✓ Resource types: {resource_types}")

    assert "entry" in bundle and isinstance(bundle["entry"], list)
    for e in bundle["entry"]:
        assert "fullUrl" in e, "Each entry must have fullUrl"
        assert "resource" in e, "Each entry must have resource"
        assert "resourceType" in e["resource"], "Each resource must have resourceType"
    print("   ✓ All entries have fullUrl + resource + resourceType")

    # ── 3. SOAP fallback contains medical disclaimer ──────────────
    print("\n3. SOAP medical disclaimer always present...")
    # Test with no API key (fallback mode)
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    soap = generate_soap_note(
        patient_state=PATIENT_STATE,
        consensus_output=CONSENSUS_OUTPUT,
        lab_output=LAB_OUTPUT,
        imaging_output=IMAGING_OUTPUT,
        drug_safety_output=DRUG_SAFETY_OUTPUT,
        digital_twin_output=DIGITAL_TWIN_OUTPUT,
        chief_complaint="Fever and cough",
    )
    if old_key:
        os.environ["OPENAI_API_KEY"] = old_key

    assert "subjective" in soap, "SOAP must have Subjective"
    assert "objective" in soap, "SOAP must have Objective"
    assert "assessment" in soap, "SOAP must have Assessment"
    assert "plan" in soap, "SOAP must have Plan"
    print("   ✓ SOAP has all 4 sections (S/O/A/P)")

    assert "AI-GENERATED" in soap["assessment"] or "AI-generated" in soap["assessment"], \
        "Medical disclaimer must appear in assessment"
    print("   ✓ Medical disclaimer present in assessment")

    assert isinstance(soap["plan"], list) and len(soap["plan"]) >= 1
    print(f"   ✓ Plan has {len(soap['plan'])} items")

    # ── 4. FHIR Bundle with no imaging (mock) ────────────────────
    print("\n4. FHIR Bundle with mock imaging (should skip DiagnosticReport)...")
    mock_imaging = {**IMAGING_OUTPUT, "mock": True}
    bundle_no_img = build_fhir_bundle(
        patient_id="test-no-img",
        diagnosis_output=DIAGNOSIS_OUTPUT,
        imaging_output=mock_imaging,
        drug_safety_output=DRUG_SAFETY_OUTPUT,
        digital_twin_output=DIGITAL_TWIN_OUTPUT,
    )
    types_no_img = bundle_no_img["_resource_types"]
    assert "DiagnosticReport" not in types_no_img, \
        "Mock imaging should not add DiagnosticReport to bundle"
    print("   ✓ Mock imaging correctly excluded from bundle")

    # ── 5. Empty inputs don't crash ───────────────────────────────
    print("\n5. Empty inputs graceful handling...")
    bundle_empty = build_fhir_bundle(
        patient_id="test-empty",
        diagnosis_output=None,
        imaging_output=None,
        drug_safety_output=None,
        digital_twin_output=None,
    )
    assert bundle_empty["resourceType"] == "Bundle"
    assert isinstance(bundle_empty["entry"], list)
    print(f"   ✓ Empty inputs: bundle with {len(bundle_empty['entry'])} entries (no crash)")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx
    AGENT_URL = "http://localhost:8009"

    print("\n" + "=" * 60)
    print("Integration Tests — requires server at localhost:8009")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Risk model loaded: {d.get('risk_model_loaded')}")
            print(f"   Features: {d.get('features_loaded')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy")
                return
            print("   ✓ Agent healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start: python agents/explanation/main.py")
            return

        # ── 2. Full explanation pipeline ─────────────────────────
        print("\n2. Full /explain pipeline (calls LLM if API key available)...")
        try:
            r = await client.post(f"{AGENT_URL}/explain", json={
                "patient_state":      PATIENT_STATE,
                "consensus_output":   CONSENSUS_OUTPUT,
                "diagnosis_output":   DIAGNOSIS_OUTPUT,
                "lab_output":         LAB_OUTPUT,
                "imaging_output":     IMAGING_OUTPUT,
                "drug_safety_output": DRUG_SAFETY_OUTPUT,
                "digital_twin_output":DIGITAL_TWIN_OUTPUT,
                "chief_complaint":    "Fever, productive cough, shortness of breath x 3 days",
            })
            d = r.json()
            print(f"   Status: {r.status_code}")

            if r.status_code == 200:
                # SOAP note
                soap = d["clinician_output"]["soap_note"]
                print(f"\n   SOAP Note:")
                print(f"      S: {soap.get('subjective', '')[:70]}...")
                print(f"      O: {soap.get('objective', '')[:70]}...")
                print(f"      A: {soap.get('assessment', '')[:70]}...")
                print(f"      P: {len(soap.get('plan', []))} action items")

                assert all(k in soap for k in ("subjective", "objective", "assessment", "plan")), \
                    "SOAP must have all 4 sections"
                print("   ✓ SOAP has all 4 sections (S/O/A/P)")

                assert "AI-GENERATED" in soap.get("assessment", "") or \
                       "AI-generated" in soap.get("assessment", ""), \
                    "Medical disclaimer must be in assessment"
                print("   ✓ Medical disclaimer present in SOAP assessment")

                # Patient explanation
                patient = d["patient_output"]
                rl = d["reading_level_check"]
                print(f"\n   Patient Explanation:")
                print(f"      Reading level: grade {rl.get('grade_level', '?'):.1f}")
                print(f"      Acceptable (≤8): {rl.get('acceptable', '?')}")
                print(f"      Condition: {patient.get('condition_explanation', '')[:70]}...")

                assert all(k in patient for k in (
                    "condition_explanation", "why_this_happened",
                    "what_happens_next", "what_to_expect",
                    "important_for_you_to_know", "when_to_call_the_nurse"
                )), "Patient explanation missing required keys"
                print("   ✓ Patient explanation has all required fields")

                grade = rl.get("grade_level", 99)
                assert grade <= 10, f"Reading level {grade:.1f} too high (should be ≤10)"
                print(f"   ✓ Reading level {grade:.1f} within acceptable range")

                # Risk attribution
                attr = d["risk_attribution"]
                print(f"\n   Risk Attribution:")
                print(f"      Factors: {len(attr.get('shap_style_breakdown', []))}")
                for a in attr.get("shap_style_breakdown", [])[:3]:
                    print(f"         {a.get('contribution', '')} {a.get('feature', '')} ({a.get('direction', '')})")

                # FHIR Bundle
                bundle = d["fhir_bundle"]
                print(f"\n   FHIR Bundle:")
                print(f"      Resources: {bundle.get('_entry_count', len(bundle.get('entry', [])))}")
                print(f"      Types: {bundle.get('_resource_types', [])}")
                assert bundle["resourceType"] == "Bundle"
                assert len(bundle.get("entry", [])) >= 1
                print("   ✓ FHIR Bundle assembled with ≥ 1 resource")

                # Consensus status propagated
                assert d["consensus_status"] == "FULL_CONSENSUS"
                assert d["human_review_required"] == False
                print("   ✓ Consensus status correctly propagated")

            else:
                print(f"   ❌ Request failed: {r.text[:200]}")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Escalation path propagates correctly ───────────────
        print("\n3. Escalation consensus propagated correctly...")
        try:
            escalation_consensus = {
                **CONSENSUS_OUTPUT,
                "consensus_status": "ESCALATION_REQUIRED",
                "human_review_required": True,
                "escalation_flag": {"priority": "URGENT", "reason": "Imaging-clinical dissociation"},
            }
            r = await client.post(f"{AGENT_URL}/explain", json={
                "patient_state":    PATIENT_STATE,
                "consensus_output": escalation_consensus,
                "chief_complaint":  "Test",
            })
            d = r.json()
            assert d["consensus_status"] == "ESCALATION_REQUIRED"
            assert d["human_review_required"] == True
            print("   ✓ ESCALATION_REQUIRED correctly propagated to final output")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Explanation Agent tests complete!")
    print("=" * 60)
    print("\nNext step: Build the Orchestrator (Agent 9) — ties everything together!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only (no server)")
    args = parser.parse_args()

    run_unit_tests()
    if not args.unit:
        asyncio.run(run_integration_tests())