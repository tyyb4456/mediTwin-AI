"""
Test script for Digital Twin Agent
Port: 8006

Part 1 — Unit tests (no server needed):
    - Feature engineering from PatientState
    - Treatment effect simulation
    - Scenario ranking

Part 2 — Integration tests (server must be running):
    - Full simulation pipeline
    - Treatment comparison: Option B > Option A > Option C
    - FHIR CarePlan structure

Run:
    # Unit tests only
    python agents/digital_twin/test.py --unit

    # Full test (needs server + trained models)
    python agents/digital_twin/train_models.py   # train first
    python agents/digital_twin/main.py            # start server
    python agents/digital_twin/test.py            # run tests
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Sample patient state for tests
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PATIENT = {
    "patient_id": "test-dt-001",
    "demographics": {"name": "John Test", "age": 54, "gender": "male", "dob": "1970-03-14"},
    "active_conditions": [
        {"code": "J18.9", "display": "Pneumonia"},
        {"code": "E11.9", "display": "Type 2 Diabetes"},
        {"code": "I48.0", "display": "Atrial fibrillation"},
    ],
    "medications": [
        {"drug": "Warfarin 5mg", "dose": "5mg", "frequency": "OD", "status": "active"},
        {"drug": "Metformin 850mg", "dose": "850mg", "frequency": "BID", "status": "active"},
    ],
    "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}],
    "lab_results": [
        {"loinc": "26464-8", "display": "WBC", "value": 18.4, "unit": "10*3/uL", "flag": "CRITICAL"},
        {"loinc": "1988-5",  "display": "CRP", "value": 142.0, "unit": "mg/L",   "flag": "HIGH"},
        {"loinc": "2160-0",  "display": "Creatinine", "value": 1.1, "unit": "mg/dL", "flag": "NORMAL"},
        {"loinc": "1751-7",  "display": "Albumin", "value": 3.6, "unit": "g/dL", "flag": "NORMAL"},
        {"loinc": "718-7",   "display": "Hemoglobin", "value": 13.8, "unit": "g/dL", "flag": "NORMAL"},
        {"loinc": "2823-3",  "display": "Potassium", "value": 4.1, "unit": "mEq/L", "flag": "NORMAL"},
    ],
    "diagnostic_reports": [], "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z", "imaging_available": True,
}

TREATMENT_OPTIONS = [
    {
        "option_id": "A",
        "label": "Azithromycin outpatient",
        "drugs": ["Azithromycin 500mg"],
        "interventions": ["O2 supplementation"],
    },
    {
        "option_id": "B",
        "label": "Ceftriaxone IV + Azithromycin (hospitalization)",
        "drugs": ["Ceftriaxone 1g IV", "Azithromycin 500mg"],
        "interventions": ["Hospitalization", "IV fluids", "Continuous monitoring"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from feature_engineering import engineer_features, FEATURE_NAMES
    from simulator import (
        simulate_treatment,
        determine_patient_risk_profile,
        select_recommended_option,
    )

    print("=" * 60)
    print("Unit Tests — feature_engineering.py + simulator.py")
    print("=" * 60)

    # ── 1. Feature engineering ────────────────────────────────────
    print("\n1. Feature engineering from PatientState...")
    fv, fd = engineer_features(SAMPLE_PATIENT)

    assert len(fv) == len(FEATURE_NAMES), \
        f"Expected {len(FEATURE_NAMES)} features, got {len(fv)}"
    print(f"   ✓ Feature vector length: {len(fv)} ({len(FEATURE_NAMES)} expected)")

    assert all(isinstance(v, (int, float)) for v in fv), \
        "All features must be numeric"
    print("   ✓ All features are numeric (int/float)")

    # Verify key features extracted correctly
    assert fd["age"] == 54.0
    print(f"   ✓ Age extracted: {fd['age']}")

    assert fd["gender_male"] == 1.0
    print(f"   ✓ Gender (male=1): {fd['gender_male']}")

    assert fd["wbc"] == 18.4
    print(f"   ✓ WBC from LOINC 26464-8: {fd['wbc']}")

    assert fd["creatinine"] == 1.1
    print(f"   ✓ Creatinine from LOINC 2160-0: {fd['creatinine']}")

    assert fd["has_diabetes"] == 1.0
    print(f"   ✓ Diabetes flag (E11.9): {fd['has_diabetes']}")

    assert fd["has_atrial_fibrillation"] == 1.0
    print(f"   ✓ Atrial fibrillation flag (I48.0): {fd['has_atrial_fibrillation']}")

    assert fd["on_anticoagulant"] == 1.0
    print(f"   ✓ Anticoagulant flag (Warfarin): {fd['on_anticoagulant']}")

    assert fd["critical_lab_count"] == 1.0   # WBC CRITICAL
    print(f"   ✓ Critical lab count: {fd['critical_lab_count']}")

    # ── 2. Treatment simulation ───────────────────────────────────
    print("\n2. Treatment simulation...")
    baseline = {"readmission_30d": 0.35, "mortality_30d": 0.12, "complication": 0.40}

    # Option A: Azithromycin alone
    pred_a = simulate_treatment(
        baseline_risks=baseline,
        drugs=["Azithromycin 500mg"],
        interventions=["O2 supplementation"],
    )

    # Option B: Ceftriaxone IV + Azithromycin + hospitalization (more aggressive)
    pred_b = simulate_treatment(
        baseline_risks=baseline,
        drugs=["Ceftriaxone 1g IV", "Azithromycin 500mg"],
        interventions=["Hospitalization", "IV fluids", "Continuous monitoring"],
    )

    # Option C: No treatment
    pred_c = {"readmission_risk_30d": 0.35, "mortality_risk_30d": 0.12, "complication_risk": 0.40,
               "recovery_probability_7d": 0.55, "estimated_recovery_days": None}

    print(f"   Option A: mortality={pred_a['mortality_risk_30d']:.2%}, recovery={pred_a['recovery_probability_7d']:.2%}")
    print(f"   Option B: mortality={pred_b['mortality_risk_30d']:.2%}, recovery={pred_b['recovery_probability_7d']:.2%}")
    print(f"   Option C (no tx): mortality={pred_c['mortality_risk_30d']:.2%}")

    # B should be better than A (more aggressive treatment)
    assert pred_b["mortality_risk_30d"] < pred_a["mortality_risk_30d"], \
        "IV treatment (B) should reduce mortality more than oral (A)"
    print("   ✓ Option B has lower mortality than Option A (more aggressive treatment)")

    assert pred_b["recovery_probability_7d"] > pred_a["recovery_probability_7d"], \
        "Option B should have higher recovery probability"
    print("   ✓ Option B has higher 7-day recovery probability")

    # Treatment must be better than no treatment
    assert pred_a["mortality_risk_30d"] < pred_c["mortality_risk_30d"], \
        "Any treatment should reduce mortality vs no treatment"
    print("   ✓ Any treatment reduces mortality vs no treatment")

    # All values must be in [0, 1]
    for pred in [pred_a, pred_b]:
        for k, v in pred.items():
            if isinstance(v, float):
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1] range"
    print("   ✓ All risk values in valid [0.0, 1.0] range")

    # ── 3. Risk profile classification ───────────────────────────
    print("\n3. Risk profile classification...")
    low_risk = {"readmission_30d": 0.05, "mortality_30d": 0.02, "complication": 0.08}
    high_risk = {"readmission_30d": 0.60, "mortality_30d": 0.30, "complication": 0.55}

    profile_low  = determine_patient_risk_profile(low_risk)
    profile_high = determine_patient_risk_profile(high_risk)

    assert profile_low in ("LOW", "MODERATE"), f"Expected LOW/MODERATE, got {profile_low}"
    assert profile_high in ("HIGH", "MODERATE-HIGH"), f"Expected HIGH/MODERATE-HIGH, got {profile_high}"
    print(f"   ✓ Low risk → {profile_low}")
    print(f"   ✓ High risk → {profile_high}")

    # ── 4. Scenario ranking ───────────────────────────────────────
    print("\n4. Scenario ranking...")
    scenarios_for_rank = [
        {"option_id": "A", "label": "Oral therapy",   "predictions": {"mortality_risk_30d": 0.08, "readmission_risk_30d": 0.24, "complication_risk": 0.25}},
        {"option_id": "B", "label": "IV therapy",     "predictions": {"mortality_risk_30d": 0.03, "readmission_risk_30d": 0.09, "complication_risk": 0.12}},
    ]
    best_id, confidence = select_recommended_option(scenarios_for_rank)
    assert best_id == "B", f"Option B (lower risk) should be recommended, got {best_id}"
    assert 0.5 <= confidence <= 1.0, f"Confidence should be in [0.5, 1.0], got {confidence}"
    print(f"   ✓ B correctly recommended (lower risk), confidence={confidence:.2f}")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx
    AGENT_URL = "http://localhost:8006"

    print("\n" + "=" * 60)
    print("Integration Tests — requires server at localhost:8006")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=45.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Models loaded: {d.get('models_loaded')}")
            print(f"   Models: {d.get('models')}")
            print(f"   LLM ready: {d.get('llm_ready')}")
            if not d.get("models_loaded"):
                print("   ⚠️  Models not loaded. Run: python train_models.py")
            if r.status_code != 200:
                print("   ❌ Agent not running.")
                return
            print("   ✓ Agent healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start: python agents/digital_twin/main.py")
            return

        # ── 2. Full simulation ───────────────────────────────────
        print("\n2. Full simulation — pneumonia patient, 2 treatment options...")
        try:
            r = await client.post(f"{AGENT_URL}/simulate", json={
                "patient_state": SAMPLE_PATIENT,
                "diagnosis": "Community-acquired pneumonia (J18.9)",
                "treatment_options": TREATMENT_OPTIONS,
            })
            d = r.json()
            print(f"   Status: {r.status_code}")

            if r.status_code == 200:
                summary = d["simulation_summary"]
                print(f"   Risk profile: {summary['patient_risk_profile']}")
                print(f"   Baseline risks: {summary.get('baseline_risks', {})}")
                print(f"   Recommended: Option {summary['recommended_option']}")
                print(f"   Confidence: {summary['recommendation_confidence']:.2f}")
                print(f"\n   Scenarios ({len(d['scenarios'])}):")
                for s in d["scenarios"]:
                    preds = s.get("predictions", {})
                    print(f"      [{s['option_id']}] {s['label']}")
                    print(f"           7d recovery: {preds.get('recovery_probability_7d', 0):.1%}, "
                          f"mortality: {preds.get('mortality_risk_30d', 0):.1%}, "
                          f"readmission: {preds.get('readmission_risk_30d', 0):.1%}")
                print(f"\n   Narrative: {d['what_if_narrative'][:120]}...")
                print(f"\n   Feature attribution ({len(d['feature_attribution'])} factors):")
                for attr in d["feature_attribution"][:3]:
                    print(f"      {attr['contribution']} {attr['feature']} ({attr['direction']})")

                # Assertions
                assert not d["mock"], "Should not be mock mode when models are loaded"
                print("\n   ✓ Not mock mode")

                assert len(d["scenarios"]) >= 2, "Should have at least 2 scenarios (A + C baseline)"
                print(f"   ✓ {len(d['scenarios'])} scenarios generated")

                # B (IV+hospitalization) should have better outcomes than A (oral outpatient)
                scene_b = next((s for s in d["scenarios"] if s["option_id"] == "B"), None)
                scene_a = next((s for s in d["scenarios"] if s["option_id"] == "A"), None)
                scene_c = next((s for s in d["scenarios"] if s["option_id"] == "C"), None)

                if scene_a and scene_b:
                    assert scene_b["predictions"]["mortality_risk_30d"] < scene_a["predictions"]["mortality_risk_30d"], \
                        "Option B (IV) should have lower mortality than A (oral)"
                    print("   ✓ Option B mortality < Option A mortality (IV > oral)")

                if scene_c and scene_a:
                    assert scene_a["predictions"]["mortality_risk_30d"] < scene_c["predictions"]["mortality_risk_30d"], \
                        "Any treatment should be better than no treatment"
                    print("   ✓ Any treatment reduces mortality vs no treatment")

                if scene_b:
                    assert summary["recommended_option"] in ("A", "B"), \
                        "Should recommend A or B, not no-treatment"
                    print(f"   ✓ Recommendation is a treatment option ({summary['recommended_option']})")

                # FHIR CarePlan
                cp = d.get("fhir_care_plan")
                if cp:
                    assert cp["resourceType"] == "CarePlan"
                    assert "subject" in cp
                    print("   ✓ FHIR CarePlan structure valid")
                else:
                    print("   ⚠️  No FHIR CarePlan in response")

            else:
                print(f"   ❌ Request failed: {r.text[:200]}")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Default options test ──────────────────────────────
        print("\n3. Simulation with no treatment options (should auto-add C baseline)...")
        try:
            r = await client.post(f"{AGENT_URL}/simulate", json={
                "patient_state": SAMPLE_PATIENT,
                "diagnosis": "Pneumonia",
                "treatment_options": [],
            })
            d = r.json()
            print(f"   Status: {r.status_code}")
            if r.status_code == 200:
                assert any(s["option_id"] == "C" for s in d["scenarios"]), \
                    "Should auto-add baseline C scenario"
                print("   ✓ Baseline 'no treatment' scenario auto-added")
        except Exception as e:
            print(f"   ⚠️  {e}")

    print("\n" + "=" * 60)
    print("Digital Twin Agent tests complete!")
    print("=" * 60)
    print("\nNext step: Build Consensus + Escalation Agent (Agent 7 — LangGraph node)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only (no server needed)")
    args = parser.parse_args()

    run_unit_tests()

    if not args.unit:
        asyncio.run(run_integration_tests())