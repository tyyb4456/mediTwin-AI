"""
Test script for Consensus + Escalation Agent
Port: 8007

Part 1 — Unit tests (no server, no API key):
    - Conflict detection: all 3 conflict types
    - Routing logic
    - Aggregate confidence scoring
    - ICD-10 compatibility check
    - Full consensus pipeline (all agents agree)
    - Conflict → escalation pipeline

Part 2 — Integration tests (server must be running):
    - REST /consensus endpoint
    - Full consensus (no conflicts)
    - Lab-diagnosis mismatch (→ resolve or escalate)
    - Drug safety conflict
    - Imaging-clinical dissociation

Run:
    python agents/consensus/test.py --unit       # Unit tests only
    python agents/consensus/test.py              # Full suite
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# ── Sample agent outputs for testing ──────────────────────────────────────────

DIAGNOSIS_PNEUMONIA = {
    "differential_diagnosis": [
        {"rank": 1, "icd10_code": "J18.9", "display": "Community-acquired pneumonia",
         "confidence": 0.87, "supporting_evidence": ["Fever", "WBC elevated", "Cough"]}
    ],
    "top_diagnosis": "Community-acquired pneumonia",
    "top_icd10_code": "J18.9",
    "confidence_level": "HIGH",
    "reasoning_summary": "Classic CAP presentation.",
    "recommended_next_steps": ["Chest X-ray", "Blood cultures"],
}

DIAGNOSIS_VIRAL = {
    "differential_diagnosis": [
        {"rank": 1, "icd10_code": "J12.9", "display": "Viral pneumonia",
         "confidence": 0.72, "supporting_evidence": ["Bilateral infiltrates"]}
    ],
    "top_diagnosis": "Viral pneumonia",
    "top_icd10_code": "J12.9",
    "confidence_level": "MODERATE",
    "reasoning_summary": "Viral etiology suspected.",
    "recommended_next_steps": ["Supportive care"],
}

LAB_CONFIRMS_PNEUMONIA = {
    "lab_summary": {"total_results": 5, "abnormal_count": 2, "critical_count": 1, "overall_severity": "SEVERE"},
    "flagged_results": [{"loinc": "26464-8", "display": "WBC", "value": 18.4, "flag": "CRITICAL"}],
    "pattern_analysis": {"identified_patterns": [{"pattern": "Bacterial infection markers"}]},
    "diagnosis_confirmation": {
        "proposed_diagnosis": "Community-acquired pneumonia",
        "proposed_icd10": "J18.9",
        "confirms_top_diagnosis": True,
        "lab_confidence_boost": 0.12,
        "alternative_diagnosis_code": None,
        "reasoning": "WBC + CRP pattern strongly supports bacterial pneumonia.",
    },
    "critical_alerts": [{"level": "CRITICAL", "message": "WBC 18.4 — sepsis workup", "action_required": True}],
}

LAB_DISAGREES_VIRAL = {
    "lab_summary": {"total_results": 5, "abnormal_count": 1, "critical_count": 0, "overall_severity": "MILD"},
    "flagged_results": [],
    "pattern_analysis": {"identified_patterns": []},
    "diagnosis_confirmation": {
        "proposed_diagnosis": "Community-acquired pneumonia",
        "proposed_icd10": "J18.9",
        "confirms_top_diagnosis": False,
        "lab_confidence_boost": 0.0,
        "alternative_diagnosis_code": "J12.9",   # Viral — different 3-char prefix from J18
        "reasoning": "WBC is normal — viral etiology more likely than bacterial.",
    },
    "critical_alerts": [],
}

IMAGING_CONFIRMS = {
    "model_output": {"prediction": "PNEUMONIA", "confidence": 0.923,
                     "pneumonia_probability": 0.923, "normal_probability": 0.077},
    "severity_assessment": {"grade": "MODERATE", "triage_priority": 2, "triage_label": "URGENT"},
    "imaging_findings": {"pattern": "Lobar consolidation", "affected_area": "Right lower lobe"},
    "clinical_interpretation": "Findings consistent with pneumonia (92.3% confidence).",
    "confirms_diagnosis": True,
    "diagnosis_code": "J18.9",
    "recommended_actions": ["Start antibiotics", "Blood cultures"],
    "fhir_diagnostic_report": None,
    "model_loaded": True,
    "mock": False,
}

IMAGING_NORMAL_BUT_CLINICAL_SUSPICIOUS = {
    "model_output": {"prediction": "NORMAL", "confidence": 0.78,
                     "pneumonia_probability": 0.22, "normal_probability": 0.78},
    "severity_assessment": {"grade": "NORMAL", "triage_priority": 4, "triage_label": "ROUTINE"},
    "imaging_findings": {"pattern": "No consolidation", "affected_area": "N/A"},
    "clinical_interpretation": "No significant consolidation detected.",
    "confirms_diagnosis": False,
    "diagnosis_code": None,
    "recommended_actions": ["Clinical correlation required"],
    "fhir_diagnostic_report": None,
    "model_loaded": True,
    "mock": False,
}

DRUG_SAFETY_SAFE = {
    "safety_status": "SAFE",
    "critical_interactions": [],
    "contraindications": [],
    "alternatives": [],
    "approved_medications": ["Azithromycin 500mg"],
    "flagged_medications": [],
}

DRUG_SAFETY_UNSAFE = {
    "safety_status": "UNSAFE",
    "critical_interactions": [],
    "contraindications": [
        {
            "drug": "Amoxicillin",
            "allergen": "Penicillin",
            "severity": "CRITICAL",
            "reason": "Patient has Penicillin allergy — Amoxicillin is cross-reactive.",
            "recommendation": "AVOID — use macrolide instead.",
        }
    ],
    "alternatives": [{"drug": "Azithromycin 500mg", "rationale": "Safe in penicillin allergy"}],
    "approved_medications": [],
    "flagged_medications": ["Amoxicillin"],
}

PATIENT_STATE = {
    "patient_id": "test-consensus-001",
    "demographics": {"name": "John Test", "age": 54, "gender": "male"},
    "active_conditions": [{"code": "J18.9", "display": "Pneumonia"}],
    "medications": [{"drug": "Warfarin", "dose": "5mg", "status": "active"}],
    "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}],
    "lab_results": [
        {"loinc": "26464-8", "display": "WBC", "value": 18.4, "unit": "10*3/uL", "flag": "CRITICAL"}
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from conflict_detector import (
        detect_conflicts, route_consensus, get_max_severity,
        _codes_compatible, _icd_prefix, Conflict,
    )
    from main import compute_aggregate_confidence, run_consensus

    print("=" * 60)
    print("Unit Tests — conflict_detector.py + main.py")
    print("=" * 60)

    # ── 1. ICD-10 prefix utilities ────────────────────────────────
    print("\n1. ICD-10 prefix utilities...")
    assert _icd_prefix("J18.9") == "J18"
    assert _icd_prefix("J12.9") == "J12"
    assert _icd_prefix("A41.9") == "A41"
    print("   ✓ _icd_prefix() correct")

    assert _codes_compatible("J18.9", "J18.1")   # Same category — compatible
    assert not _codes_compatible("J18.9", "J12.9")  # Different category — conflict
    assert _codes_compatible("", "J18.9")            # Missing code — no conflict
    print("   ✓ _codes_compatible() correct")

    # ── 2. No conflict — all agree ────────────────────────────────
    print("\n2. Full consensus — all agents agree on J18.9...")
    conflicts = detect_conflicts(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_CONFIRMS,
        drug_safety_output=DRUG_SAFETY_SAFE,
    )
    assert len(conflicts) == 0, f"Expected no conflicts, got {conflicts}"
    route = route_consensus(conflicts)
    assert route == "no_conflict"
    print("   ✓ 0 conflicts detected")
    print("   ✓ Route → no_conflict")

    # ── 3. Diagnosis-Lab mismatch ─────────────────────────────────
    print("\n3. Diagnosis-Lab mismatch (J18.9 vs J12.9)...")
    conflicts = detect_conflicts(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,  # says J18.9 (bacterial)
        lab_output=LAB_DISAGREES_VIRAL,        # says J12.9 (viral)
        imaging_output=None,
        drug_safety_output=None,
    )
    assert len(conflicts) == 1, f"Expected 1 conflict, got {len(conflicts)}"
    assert conflicts[0].type == "diagnosis_lab_mismatch"
    assert conflicts[0].severity in ("LOW", "MODERATE", "HIGH")
    print(f"   ✓ diagnosis_lab_mismatch detected (severity: {conflicts[0].severity})")
    print(f"   ✓ Description: {conflicts[0].description[:70]}...")

    route = route_consensus(conflicts)
    assert route in ("resolve", "escalate"), f"Expected resolve/escalate, got {route}"
    print(f"   ✓ Route → {route}")

    # ── 4. Imaging-clinical dissociation → always escalate ────────
    print("\n4. Imaging-clinical dissociation (imaging NORMAL, clinical HIGH confidence)...")
    conflicts = detect_conflicts(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,               # high confidence J18.9
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_NORMAL_BUT_CLINICAL_SUSPICIOUS,  # imaging says NORMAL
        drug_safety_output=None,
    )
    imaging_conflicts = [c for c in conflicts if c.type == "imaging_clinical_dissociation"]
    assert len(imaging_conflicts) >= 1, f"Expected imaging_clinical_dissociation, got {conflicts}"
    assert imaging_conflicts[0].severity == "HIGH"
    print(f"   ✓ imaging_clinical_dissociation detected (severity: HIGH)")

    route = route_consensus(conflicts)
    assert route == "escalate", f"Imaging dissociation should always escalate, got {route}"
    print("   ✓ Route → escalate (correct — imaging dissociation always escalates)")

    # ── 5. Drug safety conflict ───────────────────────────────────
    print("\n5. Drug safety conflict (Amoxicillin + Penicillin allergy)...")
    conflicts = detect_conflicts(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_CONFIRMS,
        drug_safety_output=DRUG_SAFETY_UNSAFE,
    )
    drug_conflicts = [c for c in conflicts if c.type == "treatment_contraindicated"]
    assert len(drug_conflicts) >= 1, f"Expected treatment_contraindicated, got {conflicts}"
    assert drug_conflicts[0].severity in ("MODERATE", "HIGH")
    print(f"   ✓ treatment_contraindicated detected (severity: {drug_conflicts[0].severity})")
    print(f"   ✓ Flagged medications: {drug_conflicts[0].extra.get('flagged_medications', [])}")

    # ── 6. Aggregate confidence scoring ──────────────────────────
    print("\n6. Aggregate confidence computation...")
    # All agree, high confidence
    conf = compute_aggregate_confidence(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,   # confidence 0.87
        lab_output=LAB_CONFIRMS_PNEUMONIA,      # boost 0.12
        imaging_output=IMAGING_CONFIRMS,         # confidence 0.923, confirms
        resolution=None,
    )
    assert 0.30 <= conf <= 0.99, f"Confidence out of range: {conf}"
    print(f"   ✓ Full consensus confidence: {conf:.2%} (range [0.30, 0.99])")

    # No imaging
    conf_no_img = compute_aggregate_confidence(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=None,
        resolution=None,
    )
    assert conf_no_img < conf, "Without imaging, confidence should be lower"
    print(f"   ✓ Without imaging: {conf_no_img:.2%} (correctly lower)")

    # With tiebreaker resolution bonus
    conf_resolved = compute_aggregate_confidence(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_CONFIRMS,
        resolution={"resolved_diagnosis": "J18.9"},
    )
    assert conf_resolved >= conf, "Resolution bonus should add confidence"
    print(f"   ✓ With tiebreaker bonus: {conf_resolved:.2%} (correctly higher)")

    assert conf_resolved <= 0.99, "Confidence must never reach 1.0"
    print("   ✓ Confidence capped at 0.99")

    # ── 7. Full consensus pipeline ────────────────────────────────
    print("\n7. Full consensus pipeline — FULL_CONSENSUS path...")
    result = run_consensus(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_CONFIRMS,
        drug_safety_output=DRUG_SAFETY_SAFE,
        patient_state=PATIENT_STATE,
    )
    assert result["consensus_status"] == "FULL_CONSENSUS"
    assert result["human_review_required"] == False
    assert result["conflict_count"] == 0
    assert result["aggregate_confidence"] > 0.50
    assert result["partial_outputs_available"] == True
    print(f"   ✓ status: FULL_CONSENSUS")
    print(f"   ✓ human_review_required: False")
    print(f"   ✓ aggregate_confidence: {result['aggregate_confidence']:.2%}")
    print(f"   ✓ final_diagnosis: {result['final_diagnosis']}")

    # ── 8. Escalation pipeline ────────────────────────────────────
    print("\n8. Consensus pipeline — ESCALATION path (imaging dissociation)...")
    result = run_consensus(
        diagnosis_output=DIAGNOSIS_PNEUMONIA,
        lab_output=LAB_CONFIRMS_PNEUMONIA,
        imaging_output=IMAGING_NORMAL_BUT_CLINICAL_SUSPICIOUS,
        drug_safety_output=DRUG_SAFETY_SAFE,
        patient_state=PATIENT_STATE,
    )
    assert result["consensus_status"] == "ESCALATION_REQUIRED"
    assert result["human_review_required"] == True
    assert result["escalation_flag"] is not None
    assert len(result["escalation_flag"]["recommended_actions"]) > 0
    assert result["partial_outputs_available"] == True  # Output still available
    print("   ✓ status: ESCALATION_REQUIRED")
    print("   ✓ human_review_required: True")
    print(f"   ✓ escalation priority: {result['escalation_flag']['priority']}")
    print(f"   ✓ recommended actions: {result['escalation_flag']['recommended_actions'][:2]}")
    print("   ✓ partial_outputs_available: True (agents still produced output)")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx
    AGENT_URL = "http://localhost:8007"

    print("\n" + "=" * 60)
    print("Integration Tests — requires server at localhost:8007")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Conflict types: {d.get('conflict_types')}")
            print(f"   Tiebreaker ready: {d.get('tiebreaker_ready')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy")
                return
            print("   ✓ Agent healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start: python agents/consensus/main.py")
            return

        # ── 2. Full consensus ────────────────────────────────────
        print("\n2. REST /consensus — FULL_CONSENSUS path...")
        try:
            r = await client.post(f"{AGENT_URL}/consensus", json={
                "diagnosis_output":   DIAGNOSIS_PNEUMONIA,
                "lab_output":         LAB_CONFIRMS_PNEUMONIA,
                "imaging_output":     IMAGING_CONFIRMS,
                "drug_safety_output": DRUG_SAFETY_SAFE,
                "patient_state":      PATIENT_STATE,
            })
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Consensus status: {d['consensus_status']}")
            print(f"   Confidence: {d['aggregate_confidence']:.2%}")
            print(f"   Conflicts: {d['conflict_count']}")
            assert d["consensus_status"] == "FULL_CONSENSUS"
            assert not d["human_review_required"]
            print("   ✓ FULL_CONSENSUS, no human review required")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Drug safety conflict ──────────────────────────────
        print("\n3. REST /consensus — drug safety conflict...")
        try:
            r = await client.post(f"{AGENT_URL}/consensus", json={
                "diagnosis_output":   DIAGNOSIS_PNEUMONIA,
                "lab_output":         LAB_CONFIRMS_PNEUMONIA,
                "imaging_output":     IMAGING_CONFIRMS,
                "drug_safety_output": DRUG_SAFETY_UNSAFE,
                "patient_state":      PATIENT_STATE,
            })
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Consensus status: {d['consensus_status']}")
            print(f"   Conflicts: {d['conflict_count']}")
            for c in d["conflicts"]:
                print(f"      [{c['severity']}] {c['type']}: {c['description'][:60]}...")
            assert d["conflict_count"] >= 1
            assert any(c["type"] == "treatment_contraindicated" for c in d["conflicts"])
            print("   ✓ treatment_contraindicated conflict detected via REST")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 4. Escalation path ───────────────────────────────────
        print("\n4. REST /consensus — imaging-clinical dissociation → escalation...")
        try:
            r = await client.post(f"{AGENT_URL}/consensus", json={
                "diagnosis_output":  DIAGNOSIS_PNEUMONIA,
                "lab_output":        LAB_CONFIRMS_PNEUMONIA,
                "imaging_output":    IMAGING_NORMAL_BUT_CLINICAL_SUSPICIOUS,
                "drug_safety_output": DRUG_SAFETY_SAFE,
                "patient_state":     PATIENT_STATE,
            })
            d = r.json()
            print(f"   Consensus status: {d['consensus_status']}")
            print(f"   Human review: {d['human_review_required']}")
            if d.get("escalation_flag"):
                print(f"   Escalation priority: {d['escalation_flag']['priority']}")
                print(f"   Actions: {d['escalation_flag']['recommended_actions'][:2]}")
            assert d["consensus_status"] == "ESCALATION_REQUIRED"
            assert d["human_review_required"] == True
            assert d["partial_outputs_available"] == True
            print("   ✓ Correctly escalated, partial outputs still available")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Consensus Agent tests complete!")
    print("=" * 60)
    print("\nNext step: Build Explanation Agent (Agent 8)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only (no server)")
    args = parser.parse_args()

    run_unit_tests()
    if not args.unit:
        asyncio.run(run_integration_tests())