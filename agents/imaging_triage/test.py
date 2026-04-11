"""
Test script for Imaging Triage Agent (EfficientNetB0)
Port: 8005

Tests are split into two parts:
  Part 1 — Unit tests (no server, no model needed)
      - Preprocessing pipeline shape + dtype validation
      - EfficientNet raw pixel range check (NO /255 normalization)
      - Severity classification thresholds
      - FHIR DiagnosticReport structure
      - Base64 decode

  Part 2 — Integration tests (server must be running)
      - Health check
      - Mock mode response (no .keras file)
      - Real inference (if .keras file is present)
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

sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from inference import (
        preprocess_image,
        classify_severity,
        build_fhir_diagnostic_report,
        decode_base64_image,
        IMAGE_SIZE,
        PNEUMONIA_THRESHOLD,
        NORMALIZE_IMAGENET,
    )

    print("=" * 60)
    print("Unit Tests — EfficientNetB0 inference.py")
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

    # ── Test 2: EfficientNet raw pixel range check ────────────────
    # CRITICAL: EfficientNetB0 expects raw [0,255] — NOT normalized
    print("\n2. EfficientNet preprocessing check (NO /255 normalization)...")
    assert NORMALIZE_IMAGENET == False, \
        "NORMALIZE_IMAGENET must be False — EfficientNetB0 preprocesses internally"
    print("   ✓ NORMALIZE_IMAGENET = False (correct)")

    # Pixel values should stay in [0, 255] range — not [0, 1]
    assert preprocessed.max() > 1.0, \
        f"Pixel values look normalized (max={preprocessed.max():.3f}). EfficientNet expects raw [0,255]!"
    print(f"   ✓ Pixel range: [{preprocessed.min():.1f}, {preprocessed.max():.1f}] — raw [0,255] ✓")

    # ── Test 3: Grayscale → RGB conversion ───────────────────────
    print("\n3. Grayscale image handling...")
    gray_image = Image.new("L", (256, 256), color=128)
    preprocessed_gray = preprocess_image(gray_image)
    assert preprocessed_gray.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    print(f"   ✓ Grayscale → RGB: shape {preprocessed_gray.shape}")

    # ── Test 4: Severity thresholds ──────────────────────────────
    print("\n4. Severity classification thresholds...")
    cases = [
        (0.95, 40, "SEVERE",   1, "IMMEDIATE"),
        (0.80, 40, "MODERATE", 2, "URGENT"),
        (0.60, 40, "MILD",     3, "SEMI-URGENT"),
        (0.20, 40, "NORMAL",   4, "ROUTINE"),
    ]
    for prob, age, expected_grade, expected_priority, expected_label in cases:
        result = classify_severity(prob, age)
        assert result["grade"] == expected_grade
        assert result["triage_priority"] == expected_priority
        assert result["triage_label"] == expected_label
        print(f"   ✓ prob={prob:.2f} → {result['grade']} | priority={result['triage_priority']} ({result['triage_label']})")

    # ── Test 5: Age-based priority boost ─────────────────────────
    print("\n5. Age-based priority boost...")
    young_result = classify_severity(0.80, 35)
    elder_result = classify_severity(0.80, 70)
    assert elder_result["triage_priority"] < young_result["triage_priority"]
    print(f"   ✓ 35y prob=0.80 → priority {young_result['triage_priority']}")
    print(f"   ✓ 70y prob=0.80 → priority {elder_result['triage_priority']} (bumped)")

    peds_result = classify_severity(0.80, 3)
    assert peds_result["triage_priority"] <= young_result["triage_priority"]
    print(f"   ✓ 3y  prob=0.80 → priority {peds_result['triage_priority']} (bumped)")

    # ── Test 6: FHIR DiagnosticReport ────────────────────────────
    print("\n6. FHIR DiagnosticReport builder...")
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

    assert "EfficientNetB0" in fhir["conclusion"]
    print("   ✓ Model name in conclusion")

    assert "conclusionCode" in fhir
    assert fhir["conclusionCode"][0]["coding"][0]["code"] == "J18.9"
    print("   ✓ ICD-10 J18.9 in conclusionCode for PNEUMONIA")

    # Check model metadata extension
    ext_urls = [e["url"] for e in fhir["extension"]]
    assert any("model-name" in u for u in ext_urls)
    assert any("model-auc" in u for u in ext_urls)
    print("   ✓ Model name + AUC in FHIR extensions")

    fhir_normal = build_fhir_diagnostic_report(
        patient_id="test-456",
        prediction="NORMAL",
        confidence=0.85,
        pneumonia_prob=0.15,
        severity=classify_severity(0.15, 40),
        imaging_findings={},
    )
    assert "conclusionCode" not in fhir_normal
    print("   ✓ No ICD-10 code for NORMAL prediction")

    # ── Test 7: Base64 decode ─────────────────────────────────────
    print("\n7. Base64 image decode...")
    test_img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode()

    decoded = decode_base64_image(b64_str)
    assert decoded.size == (100, 100)
    print("   ✓ Raw base64 decode: 100x100 JPEG")

    data_url = f"data:image/jpeg;base64,{b64_str}"
    decoded_url = decode_base64_image(data_url)
    assert decoded_url.size == (100, 100)
    print("   ✓ Data URL format decode: 100x100 JPEG")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

def make_test_xray_b64(pneumonia: bool = True) -> str:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[30:226, 30:226, :] = 80
    arr[90:160, 90:160, :] = 200
    if pneumonia:
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
            print(f"   Status     : {r.status_code}")
            print(f"   Model      : {d.get('model_name', 'EfficientNetB0')}")
            print(f"   Loaded     : {d.get('model_loaded')}")
            print(f"   Model path : {d.get('model_path')}")
            if d.get("model_error"):
                print(f"   Error      : {d.get('model_error')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy")
                return
            print("   ✓ Agent healthy")
            model_loaded = d.get("model_loaded", False)
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            return

        # ── 2. X-ray analysis ────────────────────────────────────
        print("\n2. X-ray analysis (synthetic image)...")
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
            print(f"   Status      : {r.status_code}")

            if r.status_code == 200:
                print(f"   Model loaded: {d['model_loaded']}")
                print(f"   Mock mode   : {d.get('mock', False)}")
                print(f"   Prediction  : {d['model_output']['prediction']}")
                print(f"   Confidence  : {d['model_output']['confidence']:.1%}")
                print(f"   Triage      : {d['severity_assessment']['triage_label']} (priority {d['severity_assessment']['triage_priority']})")

                assert "model_output" in d
                assert d["model_output"]["prediction"] in ("PNEUMONIA", "NORMAL", "MOCK_NO_MODEL")
                print("   ✓ Prediction field valid")

                assert 1 <= d["severity_assessment"]["triage_priority"] <= 4
                print("   ✓ Triage priority in range [1-4]")

                assert len(d["recommended_actions"]) > 0
                print("   ✓ Recommended actions present")

                if d["model_loaded"] and d.get("fhir_diagnostic_report"):
                    fhir = d["fhir_diagnostic_report"]
                    assert fhir["resourceType"] == "DiagnosticReport"
                    assert "EfficientNetB0" in fhir["conclusion"]
                    print("   ✓ FHIR DiagnosticReport valid")
                    print("   ✓ EfficientNetB0 identified in FHIR conclusion")
                else:
                    print("   ℹ️  Mock mode — place efficientnet_b0.keras in models/")
            else:
                print(f"   ❌ Failed: {r.text}")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. Elderly patient ───────────────────────────────────
        print("\n3. Elderly patient (age 72)...")
        try:
            r = await client.post(
                f"{AGENT_URL}/analyze-xray",
                json={
                    "patient_id": "test-elder-002",
                    "image_data": {"format": "base64", "content_type": "image/jpeg", "data": make_test_xray_b64()},
                    "patient_context": {"age": 72, "gender": "female"},
                },
            )
            d = r.json()
            print(f"   Priority for 72y: {d['severity_assessment']['triage_priority']} ({d['severity_assessment']['triage_label']})")
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
            assert r.status_code in (400, 422)
            print(f"   ✓ Invalid base64 rejected with {r.status_code}")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ⚠️  {e}")

    print("\n" + "=" * 60)
    print("Imaging Triage Agent (EfficientNetB0) tests complete!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only")
    args = parser.parse_args()

    run_unit_tests()

    if not args.unit:
        asyncio.run(run_integration_tests())