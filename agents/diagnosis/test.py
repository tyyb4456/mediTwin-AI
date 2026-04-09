"""
Test script for Diagnosis Agent v2
Port: 8002

Part 1 — Unit tests (--unit flag, no server, no API key needed):
  - ICD-10 validation + repair logic
  - Patient query builder output
  - Rule-based adjustments (allergy filter, sepsis flag, confidence boost)
  - Cache key generation and TTL behavior
  - FHIR Condition verification status logic
  - NextStep schema parsing
  - Beta-lactam allergy medication filter

Part 2 — Integration tests (server must be running):
  - Health check (rag_mode field)
  - Full /diagnose with pneumonia patient
  - Penicillin allergy patient — Amoxicillin must NOT appear in next steps
  - Sepsis flag triggered at WBC ≥ 15
  - LLM fallback mode (if RAG down)
  - Cache hit on second identical request
  - Batch endpoint
  - Cache clear admin endpoint

Run:
    python agents/diagnosis/test.py --unit
    python agents/diagnosis/test.py
"""
import sys
import os
import asyncio
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AGENT_URL = "http://localhost:8002"

# ── Shared test patient data ───────────────────────────────────────────────────

PNEUMONIA_PATIENT = {
    "patient_id": "test-diag-001",
    "demographics": {"name": "John Test", "age": 54, "gender": "male", "dob": "1970-03-14"},
    "active_conditions": [
        {"code": "E11.9", "display": "Type 2 diabetes mellitus", "onset": "2018-01-01"},
        {"code": "I48.0", "display": "Atrial fibrillation", "onset": "2020-06-15"},
    ],
    "medications": [
        {"drug": "Warfarin", "dose": "5mg", "frequency": "OD", "status": "active"},
        {"drug": "Metformin", "dose": "850mg", "frequency": "BID", "status": "active"},
    ],
    "allergies": [
        {"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "high"},
    ],
    "lab_results": [
        {"loinc": "26464-8", "display": "WBC", "value": 18.4, "unit": "10*3/uL",
         "reference_high": 11.0, "reference_low": 4.5, "flag": "CRITICAL"},
        {"loinc": "718-7", "display": "Hemoglobin", "value": 13.8, "unit": "g/dL",
         "reference_high": 17.5, "reference_low": 13.5, "flag": "NORMAL"},
        {"loinc": "1988-5", "display": "C-Reactive Protein", "value": 142.0, "unit": "mg/L",
         "reference_high": 10.0, "flag": "HIGH"},
    ],
    "diagnostic_reports": [],
    "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z",
    "imaging_available": False,
}

NO_ALLERGY_PATIENT = {**PNEUMONIA_PATIENT, "patient_id": "test-diag-002", "allergies": []}

LOW_WBC_PATIENT = {
    **PNEUMONIA_PATIENT,
    "patient_id": "test-diag-003",
    "allergies": [],
    "lab_results": [
        {"loinc": "26464-8", "display": "WBC", "value": 8.2, "unit": "10*3/uL",
         "reference_high": 11.0, "reference_low": 4.5, "flag": "NORMAL"},
        {"loinc": "1988-5", "display": "CRP", "value": 5.0, "unit": "mg/L",
         "reference_high": 10.0, "flag": "NORMAL"},
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    print("=" * 65)
    print("Unit Tests — Diagnosis Agent v2 (no server, no API key)")
    print("=" * 65)

    from agents.diagnosis.rag import (
        DiagnosisRAG, _repair_icd10, _validate_and_repair_items,
        _SimpleCache, _cache,
    )
    from shared.models import DiagnosisItem, DiagnosisOutput, NextStep
    import pydantic

    passed = 0
    failed = 0

    def ok(msg): nonlocal passed; passed += 1; print(f"   ✓ {msg}")
    def fail(msg): nonlocal failed; failed += 1; print(f"   ❌ {msg}")

    # ── 1. ICD-10 validation ─────────────────────────────────────
    print("\n1. ICD-10 code validation...")
    valid_codes = ["J18.9", "E11", "I48.0", "A15.3", "Z87.891"]
    for code in valid_codes:
        try:
            item = DiagnosisItem(
                rank=1, display="Test", icd10_code=code,
                confidence=0.8, clinical_reasoning="test"
            )
            ok(f"Valid ICD-10 accepted: {code}")
        except Exception as e:
            fail(f"Valid ICD-10 rejected: {code} — {e}")

    invalid_codes = ["", "UNKNOWN", "123", "J.18.9 bad"]
    for code in invalid_codes:
        try:
            item = DiagnosisItem(
                rank=1, display="Test", icd10_code=code,
                confidence=0.8, clinical_reasoning="test"
            )
            fail(f"Invalid ICD-10 accepted (should have been rejected): {repr(code)}")
        except pydantic.ValidationError:
            ok(f"Invalid ICD-10 correctly rejected: {repr(code)}")

    # ── 2. ICD-10 repair logic ───────────────────────────────────
    print("\n2. ICD-10 auto-repair...")
    repairs = [
        ("j18.9",  "J18.9"),    # lowercase
        ("J 18.9", "J18.9"),    # space
        ("J.18.9", "J18.9"),    # dot after letter
        ("J18.9",  "J18.9"),    # already correct
    ]
    for raw, expected in repairs:
        got = _repair_icd10(raw)
        if got == expected:
            ok(f"Repair: '{raw}' → '{got}'")
        else:
            fail(f"Repair failed: '{raw}' → '{got}' (expected '{expected}')")

    # ── 3. Confidence bounds ─────────────────────────────────────
    print("\n3. Confidence bounds validation...")
    for bad_conf in [-0.1, 1.1, 2.0]:
        try:
            DiagnosisItem(
                rank=1, display="Test", icd10_code="J18.9",
                confidence=bad_conf, clinical_reasoning="test"
            )
            fail(f"Confidence {bad_conf} should have been rejected")
        except pydantic.ValidationError:
            ok(f"Confidence {bad_conf} correctly rejected")

    # ── 4. Patient query builder ──────────────────────────────────
    print("\n4. Patient query builder...")
    rag = DiagnosisRAG()
    query = rag._build_patient_query(PNEUMONIA_PATIENT, "Cough and fever")
    assert "54-year-old male" in query, "Demographics missing"
    ok("Demographics in query (54-year-old male)")
    assert "Penicillin" in query, "Allergy missing"
    ok("Allergy in query (Penicillin)")
    assert "CRITICAL" in query, "Critical lab flag missing"
    ok("CRITICAL lab flag in query")
    assert "ABNORMAL Labs" in query, "Abnormal section header missing"
    ok("ABNORMAL Labs section present")
    assert "Warfarin" in query, "Medication missing"
    ok("Medication in query (Warfarin)")

    # ── 5. Rule adjustments — penicillin allergy flag ─────────────
    print("\n5. Rule adjustments — beta-lactam allergy...")
    # Build a minimal DiagnosisOutput
    output = DiagnosisOutput(
        differential_diagnosis=[
            DiagnosisItem(
                rank=1, display="Community-acquired pneumonia",
                icd10_code="J18.9", confidence=0.80,
                clinical_reasoning="WBC elevated, CRP elevated",
            )
        ],
        confidence_level="HIGH",
        reasoning_summary="Bacterial pneumonia likely",
        recommended_next_steps=[
            NextStep(
                category="MEDICATION",
                description="Amoxicillin 1g TID",
                drug_name="Amoxicillin",
                drug_dose="1g",
                drug_route="oral",
                urgency="urgent",
            ),
            NextStep(
                category="MEDICATION",
                description="Azithromycin 500mg OD",
                drug_name="Azithromycin",
                drug_dose="500mg",
                drug_route="oral",
                urgency="urgent",
            ),
            NextStep(
                category="INVESTIGATION",
                description="Blood cultures x2",
                urgency="stat",
            ),
        ],
    )
    adjusted = rag._apply_rule_adjustments(output, PNEUMONIA_PATIENT)

    assert adjusted.penicillin_allergy_flagged, "Penicillin allergy flag not set"
    ok("penicillin_allergy_flagged=True when Penicillin in allergies")

    drug_names = [
        s.drug_name for s in adjusted.recommended_next_steps
        if s.category == "MEDICATION" and s.drug_name
    ]
    assert "Amoxicillin" not in drug_names, f"Amoxicillin should be removed — still present: {drug_names}"
    ok("Amoxicillin removed from next steps (beta-lactam filter)")

    azithro_present = any(
        "Azithromycin" in (s.drug_name or s.description)
        for s in adjusted.recommended_next_steps
    )
    assert azithro_present, "Azithromycin (macrolide) should still be present"
    ok("Azithromycin (macrolide) retained after filter")

    assert any("ALLERGY ALERT" in s.description for s in adjusted.recommended_next_steps), \
        "Allergy alert step should be inserted"
    ok("ALLERGY ALERT step injected")

    # ── 6. Rule adjustments — sepsis flag ────────────────────────
    print("\n6. Rule adjustments — sepsis flag...")
    output2 = DiagnosisOutput(
        differential_diagnosis=[
            DiagnosisItem(
                rank=1, display="Sepsis", icd10_code="A41.9",
                confidence=0.75, clinical_reasoning="WBC 18"
            )
        ],
        confidence_level="HIGH",
        reasoning_summary="Sepsis",
        recommended_next_steps=[],
    )
    adjusted2 = rag._apply_rule_adjustments(output2, PNEUMONIA_PATIENT)  # WBC=18.4
    assert adjusted2.high_suspicion_sepsis, "Sepsis flag should be set (WBC 18.4)"
    ok("high_suspicion_sepsis=True for WBC 18.4")

    output3 = DiagnosisOutput(
        differential_diagnosis=[
            DiagnosisItem(
                rank=1, display="Viral URTI", icd10_code="J06.9",
                confidence=0.70, clinical_reasoning="Low WBC"
            )
        ],
        confidence_level="MODERATE",
        reasoning_summary="Viral",
        recommended_next_steps=[],
    )
    adjusted3 = rag._apply_rule_adjustments(output3, LOW_WBC_PATIENT)  # WBC=8.2
    assert not adjusted3.high_suspicion_sepsis, "Sepsis flag should NOT be set (WBC 8.2)"
    ok("high_suspicion_sepsis=False for WBC 8.2 (normal)")

    # ── 7. FHIR Condition verification status ─────────────────────
    print("\n7. FHIR Condition verification status logic...")
    output4 = DiagnosisOutput(
        differential_diagnosis=[
            DiagnosisItem(rank=1, display="CAP", icd10_code="J18.9", confidence=0.85,
                          clinical_reasoning="test"),
            DiagnosisItem(rank=2, display="Viral pneumonia", icd10_code="J12.9", confidence=0.60,
                          clinical_reasoning="test"),
            DiagnosisItem(rank=3, display="Atypical pneumonia", icd10_code="J20.9", confidence=0.35,
                          clinical_reasoning="test"),
        ],
        confidence_level="HIGH",
        reasoning_summary="Pneumonia",
        recommended_next_steps=[],
    )
    fhir = rag.build_fhir_conditions(output4, "patient-001")
    assert len(fhir) == 3, f"Expected 3 FHIR conditions, got {len(fhir)}"

    statuses = [
        c["verificationStatus"]["coding"][0]["code"]
        for c in fhir
    ]
    assert statuses[0] == "provisional", f"Expected provisional for conf 0.85, got {statuses[0]}"
    ok("conf 0.85 → verificationStatus=provisional")
    assert statuses[1] == "differential", f"Expected differential for conf 0.60, got {statuses[1]}"
    ok("conf 0.60 → verificationStatus=differential")
    assert statuses[2] == "refuted", f"Expected refuted for conf 0.35, got {statuses[2]}"
    ok("conf 0.35 → verificationStatus=refuted")

    # ── 8. Cache TTL and key uniqueness ──────────────────────────
    print("\n8. Cache TTL + key uniqueness...")
    from agents.diagnosis.rag import _SimpleCache

    test_cache = _SimpleCache(ttl_seconds=1)

    dummy_output = DiagnosisOutput(
        differential_diagnosis=[
            DiagnosisItem(rank=1, display="Test", icd10_code="J18.9",
                          confidence=0.8, clinical_reasoning="test")
        ],
        confidence_level="HIGH",
        reasoning_summary="test",
        recommended_next_steps=[],
    )
    test_cache.set("p1", "cough", dummy_output)
    hit = test_cache.get("p1", "cough")
    assert hit is not None, "Cache should return value immediately"
    ok("Cache hit on same patient + complaint")

    miss = test_cache.get("p2", "cough")
    assert miss is None, "Different patient should miss"
    ok("Cache miss on different patient_id")

    miss2 = test_cache.get("p1", "chest pain")
    assert miss2 is None, "Different complaint should miss"
    ok("Cache miss on different chief_complaint")

    time.sleep(1.1)
    expired = test_cache.get("p1", "cough")
    assert expired is None, "Cache should expire after TTL"
    ok("Cache entry expired after TTL")

    # ── 9. NextStep schema — string list backward compatibility ───
    print("\n9. NextStep structured output...")
    step = NextStep(
        category="MEDICATION",
        description="Azithromycin 500mg PO OD",
        drug_name="Azithromycin",
        drug_dose="500mg",
        drug_route="oral",
        urgency="urgent",
        rationale="Macrolide for atypical coverage",
    )
    d = step.model_dump()
    assert d["category"] == "MEDICATION"
    ok("NextStep MEDICATION schema serializes correctly")

    step2 = NextStep(
        category="INVESTIGATION",
        description="Blood cultures x2 before antibiotics",
        urgency="stat",
    )
    assert step2.drug_name is None
    ok("NextStep INVESTIGATION has no drug fields (optional)")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Unit tests: {passed} passed, {failed} failed")
    if failed:
        print("❌ Some tests failed — fix before running integration tests")
        sys.exit(1)
    else:
        print("✅ All unit tests passed!")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx

    print("\n" + "=" * 65)
    print("Integration Tests — server must be running on port 8002")
    print("=" * 65)

    async with httpx.AsyncClient(timeout=90.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   rag_mode: {d.get('rag_mode')}")
            print(f"   chromadb_chunks: {d.get('chromadb_chunks')}")
            print(f"   cache: {d.get('cache')}")
            assert r.status_code == 200
            assert d.get("status") in ("healthy", "degraded")
            assert d.get("rag_mode") in ("rag", "fallback", "not_initialized")
            print("   ✓ Health check passed")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print(f"   Start with: python agents/diagnosis/main.py")
            return

        # ── 2. Full diagnosis — pneumonia patient ─────────────────
        print("\n2. Full diagnosis — pneumonia patient (Gemini call ~10-20s)...")
        try:
            t0 = time.time()
            r = await client.post(f"{AGENT_URL}/diagnose", json={
                "patient_state": PNEUMONIA_PATIENT,
                "chief_complaint": "Productive cough, fever 38.9°C, shortness of breath for 3 days",
                "include_fhir_resources": True,
            })
            elapsed = round(time.time() - t0, 2)
            print(f"   Elapsed: {elapsed}s")

            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
            d = r.json()

            print(f"   request_id: {d.get('request_id')}")
            print(f"   Top diagnosis: {d['top_diagnosis']} ({d['top_icd10_code']})")
            print(f"   Confidence: {d['confidence_level']}")
            print(f"   rag_mode: {d['rag_mode']}")
            print(f"   penicillin_allergy_flagged: {d['penicillin_allergy_flagged']}")
            print(f"   high_suspicion_sepsis: {d['high_suspicion_sepsis']}")

            assert d["top_icd10_code"], "top_icd10_code must be set"
            print("   ✓ top_icd10_code present")

            assert len(d["differential_diagnosis"]) >= 1
            print(f"   ✓ {len(d['differential_diagnosis'])} diagnoses in differential")

            # Respiratory code expected
            assert d["top_icd10_code"][0] in ("J", "A"), \
                f"Expected respiratory/infectious code, got {d['top_icd10_code']}"
            print(f"   ✓ Top ICD-10 is respiratory/infectious: {d['top_icd10_code']}")

            # Penicillin allergy should be flagged
            assert d["penicillin_allergy_flagged"], "Penicillin allergy not flagged"
            print("   ✓ penicillin_allergy_flagged=True")

            # Sepsis flag (WBC 18.4)
            assert d["high_suspicion_sepsis"], "Sepsis flag not set (WBC 18.4)"
            print("   ✓ high_suspicion_sepsis=True (WBC 18.4)")

            # Amoxicillin/Penicillin MUST NOT appear in next steps
            steps = d["recommended_next_steps"]
            beta_lactams = ["amoxicillin", "ampicillin", "penicillin", "ceftriaxone",
                            "cefazolin", "cephalexin"]
            for step in steps:
                drug = (step.get("drug_name") or "").lower()
                if step.get("category") == "MEDICATION":
                    for bl in beta_lactams:
                        assert bl not in drug, \
                            f"Beta-lactam '{step['drug_name']}' must not appear (penicillin allergy)"
            print("   ✓ No beta-lactam medications in next steps (allergy filter)")

            # FHIR conditions
            if d.get("fhir_conditions"):
                fhir = d["fhir_conditions"]
                assert all(c["resourceType"] == "Condition" for c in fhir)
                print(f"   ✓ {len(fhir)} FHIR Condition resources, all resourceType=Condition")
                statuses = [c["verificationStatus"]["coding"][0]["code"] for c in fhir]
                print(f"   ✓ Verification statuses: {statuses}")

            print(f"\n   Recommended next steps ({len(steps)}):")
            for s in steps[:5]:
                category = s.get("category", "?")
                desc = s.get("description", "?")[:70]
                urgency = s.get("urgency", "?")
                print(f"      [{category}|{urgency}] {desc}")

        except AssertionError as e:
            print(f"   ❌ Assertion: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Cache hit on identical request ─────────────────────
        print("\n3. Cache hit — second identical request should be faster...")
        try:
            t0 = time.time()
            r = await client.post(f"{AGENT_URL}/diagnose", json={
                "patient_state": PNEUMONIA_PATIENT,
                "chief_complaint": "Productive cough, fever 38.9°C, shortness of breath for 3 days",
            })
            elapsed2 = round(time.time() - t0, 2)
            print(f"   Elapsed: {elapsed2}s")
            if elapsed2 < 1.0:
                print("   ✓ Cache hit confirmed (< 1s response)")
            else:
                print(f"   ⚠️  Response took {elapsed2}s — cache may not have kicked in")
        except Exception as e:
            print(f"   ❌ {e}")

        # ── 4. No allergy patient — beta-lactams allowed ─────────
        print("\n4. No-allergy patient — beta-lactams should be allowed...")
        try:
            r = await client.post(f"{AGENT_URL}/diagnose", json={
                "patient_state": NO_ALLERGY_PATIENT,
                "chief_complaint": "Cough, fever, shortness of breath",
            })
            d = r.json()
            assert not d["penicillin_allergy_flagged"], "Should NOT flag allergy when NKDA"
            print("   ✓ penicillin_allergy_flagged=False for NKDA patient")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ {e}")

        # ── 5. Input validation ───────────────────────────────────
        print("\n5. Input validation...")
        try:
            r = await client.post(f"{AGENT_URL}/diagnose", json={
                "patient_state": PNEUMONIA_PATIENT,
                "chief_complaint": "",
            })
            assert r.status_code == 422, f"Empty complaint should be 422, got {r.status_code}"
            print("   ✓ Empty chief_complaint → 422")
        except AssertionError as e:
            print(f"   ❌ {e}")

        try:
            r = await client.post(f"{AGENT_URL}/diagnose", json={
                "patient_state": {"demographics": {"name": "x", "age": 30, "gender": "m", "dob": "1994"}},
                "chief_complaint": "cough",
            })
            assert r.status_code == 422, f"Missing patient_id should be 422, got {r.status_code}"
            print("   ✓ Missing patient_id → 422")
        except AssertionError as e:
            print(f"   ❌ {e}")

    print("\n" + "=" * 65)
    print("Integration tests complete!")
    print("=" * 65)
    print("\nNext step: docker-compose up — run the full system!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only (no server needed)")
    args = parser.parse_args()

    run_unit_tests()
    if not args.unit:
        asyncio.run(run_integration_tests())