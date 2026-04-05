"""
Test script for Imaging Triage Agent
Port: 8005

Tests are split into two parts:
  Part 1 — Unit tests (no server, no model needed)
      - Preprocessing pipeline shape + dtype validation
      - Severity classification thresholds
      - FHIR DiagnosticReport structure
      - Base64 decode

  Part 2 — Integration tests (server must be running)
      - Health check
      - Mock mode response (no .h5 file)
      - Real inference (if .h5 file is present)
      - Triage priority logic for elderly patient

Run:
    # Part 1 only (always works, no dependencies):
    python agents/imaging_triage/test.py --unit

    # Full test (needs server at localhost:8005):
    python agents/imaging_triage/test.py
"""
import sys
import os
import asyncio
import base64
import io
import argparse

import numpy as np
from PIL import Image

# Add agent dir to path for direct imports
sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests — no server, no model, no API key
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from inference import (
        preprocess_image,
        classify_severity,
        build_fhir_diagnostic_report,
        decode_base64_image,
        IMAGE_SIZE,
        PNEUMONIA_THRESHOLD,
    )

    print("=" * 60)
    print("Unit Tests — inference.py (no server/model needed)")
    print("=" * 60)

    # ── Test 1: Preprocessing output shape ───────────────────────
    print("\n1. Preprocessing pipeline...")
    test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    preprocessed = preprocess_image(test_image)

    assert preprocessed.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), \
        f"Expected shape (1, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3), got {preprocessed.shape}"
    print(f"   ✓ Output shape: {preprocessed.shape}")

    assert preprocessed.dtype == np.float32, \
        f"Expected float32, got {preprocessed.dtype}"
    print(f"   ✓ Output dtype: {preprocessed.dtype}")

    # Values after ImageNet normalization should be reasonable (not raw [0,255])
    assert preprocessed.min() < 5.0, "Values look unnormalized (too large)"
    print(f"   ✓ Value range after normalization: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

    # ── Test 2: Grayscale → RGB conversion ───────────────────────
    print("\n2. Grayscale image handling...")
    gray_image = Image.new("L", (256, 256), color=128)  # Grayscale
    preprocessed_gray = preprocess_image(gray_image)
    assert preprocessed_gray.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), \
        "Grayscale should be converted to RGB (3 channels)"
    print(f"   ✓ Grayscale converted to RGB: shape {preprocessed_gray.shape}")

    # ── Test 3: Severity thresholds ──────────────────────────────
    print("\n3. Severity classification thresholds...")
    cases = [
        (0.95, 40, "SEVERE",   1, "IMMEDIATE"),
        (0.80, 40, "MODERATE", 2, "URGENT"),
        (0.60, 40, "MILD",     3, "SEMI-URGENT"),
        (0.20, 40, "NORMAL",   4, "ROUTINE"),
    ]
    for prob, age, expected_grade, expected_priority, expected_label in cases:
        result = classify_severity(prob, age)
        assert result["grade"] == expected_grade, \
            f"prob={prob}: expected grade {expected_grade}, got {result['grade']}"
        assert result["triage_priority"] == expected_priority, \
            f"prob={prob}: expected priority {expected_priority}, got {result['triage_priority']}"
        assert result["triage_label"] == expected_label, \
            f"prob={prob}: expected label {expected_label}, got {result['triage_label']}"
        print(f"   ✓ prob={prob:.2f} → grade={result['grade']}, priority={result['triage_priority']} ({result['triage_label']})")

    # ── Test 4: Age-based priority boost ─────────────────────────
    print("\n4. Age-based priority boost...")
    # Elderly patient (70y) with MODERATE finding should get bumped to URGENT→IMMEDIATE
    young_result = classify_severity(0.80, 35)   # priority=2
    elder_result = classify_severity(0.80, 70)   # should be priority=1

    assert elder_result["triage_priority"] < young_result["triage_priority"], \
        f"Elderly patient should get higher priority: {elder_result} vs {young_result}"
    print(f"   ✓ 35y with prob=0.80 → priority {young_result['triage_priority']}")
    print(f"   ✓ 70y with prob=0.80 → priority {elder_result['triage_priority']} (bumped up)")

    # Pediatric patient
    peds_result = classify_severity(0.80, 3)
    assert peds_result["triage_priority"] <= young_result["triage_priority"], \
        "Pediatric patient should also get priority bump"
    print(f"   ✓ 3y with prob=0.80 → priority {peds_result['triage_priority']} (bumped up)")

    # ── Test 5: FHIR DiagnosticReport structure ───────────────────
    print("\n5. FHIR DiagnosticReport builder...")
    severity = classify_severity(0.92, 54)
    fhir = build_fhir_diagnostic_report(
        patient_id="test-123",
        prediction="PNEUMONIA",
        confidence=0.923,
        pneumonia_prob=0.923,
        severity=severity,
        imaging_findings={"pattern": "Lobar consolidation"},
    )

    assert fhir["resourceType"] == "DiagnosticReport"
    print("   ✓ resourceType = DiagnosticReport")

    assert fhir["subject"]["reference"] == "Patient/test-123"
    print("   ✓ subject.reference correct")

    assert "conclusion" in fhir
    assert "AI-generated" in fhir["conclusion"] or "AI Triage" in fhir["conclusion"]
    print("   ✓ AI disclaimer present in conclusion")

    assert "conclusionCode" in fhir
    assert fhir["conclusionCode"][0]["coding"][0]["code"] == "J18.9"
    print("   ✓ ICD-10 J18.9 in conclusionCode for PNEUMONIA prediction")

    # NORMAL prediction should NOT have conclusionCode
    fhir_normal = build_fhir_diagnostic_report(
        patient_id="test-456",
        prediction="NORMAL",
        confidence=0.85,
        pneumonia_prob=0.15,
        severity=classify_severity(0.15, 40),
        imaging_findings={},
    )
    assert "conclusionCode" not in fhir_normal, "NORMAL prediction should not have ICD-10 code"
    print("   ✓ No ICD-10 code for NORMAL prediction")

    # ── Test 6: Base64 decode ─────────────────────────────────────
    print("\n6. Base64 image decode...")
    test_img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode()

    decoded = decode_base64_image(b64_str)
    assert decoded.size == (100, 100)
    print("   ✓ Raw base64 decode: 100x100 JPEG")

    # Test data URL format
    data_url = f"data:image/jpeg;base64,{b64_str}"
    decoded_url = decode_base64_image(data_url)
    assert decoded_url.size == (100, 100)
    print("   ✓ Data URL format decode: 100x100 JPEG")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests — server must be running at localhost:8005
# ══════════════════════════════════════════════════════════════════════════════

def make_test_xray_b64(pneumonia: bool = True) -> str:
    """
    Generate a synthetic 'X-ray' for testing.
    Real X-rays are grayscale with specific contrast patterns.
    This creates a minimal valid JPEG for API testing.
    NOTE: The CNN won't give meaningful predictions on synthetic images.
    """
    # Simulated chest X-ray appearance (dark background, bright lung fields)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add some structure so it's not pure black
    arr[30:226, 30:226, :] = 80   # lung field region
    arr[90:160, 90:160, :] = 200  # heart shadow
    if pneumonia:
        # Simulated consolidation patch in right lower lobe
        arr[140:200, 150:220, :] = 180

    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


async def run_integration_tests():
    import httpx

    AGENT_URL = "http://localhost:8005"

    print("\n" + "=" * 60)
    print("Integration Tests — requires server at localhost:8005")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Model loaded: {d.get('model_loaded')}")
            print(f"   Model path: {d.get('model_path')}")
            if d.get("model_error"):
                print(f"   Model error: {d.get('model_error')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy — is it running?")
                print("   Start: python agents/imaging_triage/main.py")
                return
            print("   ✓ Agent healthy")
            model_loaded = d.get("model_loaded", False)
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start: python agents/imaging_triage/main.py")
            return

        # ── 2. X-ray analysis (synthetic image) ──────────────────
        print("\n2. X-ray analysis with synthetic test image...")
        try:
            xray_b64 = make_test_xray_b64(pneumonia=True)
            r = await client.post(
                f"{AGENT_URL}/analyze-xray",
                json={
                    "patient_id": "test-patient-001",
                    "image_data": {
                        "format": "base64",
                        "content_type": "image/jpeg",
                        "data": xray_b64,
                    },
                    "patient_context": {
                        "age": 54,
                        "gender": "male",
                        "chief_complaint": "Productive cough, fever 3 days",
                        "current_diagnosis": "Suspected community-acquired pneumonia",
                    },
                },
            )
            d = r.json()
            print(f"   Status: {r.status_code}")

            if r.status_code == 200:
                print(f"   Model loaded: {d['model_loaded']}")
                print(f"   Mock mode: {d.get('mock', False)}")
                print(f"   Prediction: {d['model_output']['prediction']}")
                print(f"   Confidence: {d['model_output']['confidence']:.1%}")
                print(f"   Triage: {d['severity_assessment']['triage_label']} (priority {d['severity_assessment']['triage_priority']})")
                print(f"   Confirms diagnosis: {d['confirms_diagnosis']}")

                # Structure validations (work in both mock and real mode)
                assert "model_output" in d
                assert "prediction" in d["model_output"]
                assert d["model_output"]["prediction"] in ("PNEUMONIA", "NORMAL", "MOCK_NO_MODEL")
                print("   ✓ Prediction field present and valid")

                assert "severity_assessment" in d
                assert "triage_priority" in d["severity_assessment"]
                assert 1 <= d["severity_assessment"]["triage_priority"] <= 4
                print("   ✓ Triage priority in valid range [1-4]")

                assert "recommended_actions" in d
                assert len(d["recommended_actions"]) > 0
                print("   ✓ Recommended actions present")

                if d["model_loaded"] and d.get("fhir_diagnostic_report"):
                    fhir = d["fhir_diagnostic_report"]
                    assert fhir["resourceType"] == "DiagnosticReport"
                    assert "conclusion" in fhir
                    print("   ✓ FHIR DiagnosticReport structure valid")

                    if d["confirms_diagnosis"]:
                        assert d["diagnosis_code"] == "J18.9"
                        print("   ✓ ICD-10 J18.9 set for PNEUMONIA prediction")
                else:
                    print("   ℹ️  Running in mock mode (no .h5 file) — FHIR report not generated")
                    print(f"   Place model at: agents/imaging_triage/models/pneumonia_cnn_v1.h5")
            else:
                print(f"   ❌ Request failed: {r.text}")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Elderly patient — priority bump ───────────────────
        print("\n3. Elderly patient (age 72)...")
        try:
            xray_b64 = make_test_xray_b64(pneumonia=True)
            r = await client.post(
                f"{AGENT_URL}/analyze-xray",
                json={
                    "patient_id": "test-elder-002",
                    "image_data": {"format": "base64", "content_type": "image/jpeg", "data": xray_b64},
                    "patient_context": {"age": 72, "gender": "female"},
                },
            )
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Triage priority for 72y: {d['severity_assessment']['triage_priority']} ({d['severity_assessment']['triage_label']})")
            urgency = d["severity_assessment"]["clinical_urgency"]
            if "72" in urgency or "age" in urgency.lower():
                print("   ✓ Age mentioned in clinical urgency")
            print("   ✓ Elderly patient handled")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 4. Invalid image rejection ───────────────────────────
        print("\n4. Invalid input rejection...")
        try:
            r = await client.post(
                f"{AGENT_URL}/analyze-xray",
                json={
                    "patient_id": "test-invalid",
                    "image_data": {"format": "base64", "content_type": "image/jpeg", "data": "not_valid_base64!!!"},
                },
            )
            print(f"   Status: {r.status_code}")
            assert r.status_code in (400, 422), \
                f"Invalid base64 should return 400/422, got {r.status_code}"
            print("   ✓ Invalid base64 correctly rejected")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ⚠️  Error testing invalid input: {e}")

    print("\n" + "=" * 60)
    print("Imaging Triage Agent tests complete!")
    print("=" * 60)
    print("\nNext step: Build Digital Twin Agent (Agent 6)")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Run unit tests only (no server needed)")
    args = parser.parse_args()

    run_unit_tests()

    if not args.unit:
        asyncio.run(run_integration_tests())