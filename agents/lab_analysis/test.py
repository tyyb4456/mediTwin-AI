"""
Test script for Lab Analysis Agent
Tests at http://localhost:8003

The rules engine runs entirely without OpenAI — so most tests work without an API key.
The LLM interpretation layer needs OPENAI_API_KEY, but the agent degrades gracefully without it.

Run:
    python agents/lab_analysis/test.py

Prerequisites:
    python agents/lab_analysis/main.py   (in another terminal)
"""
import asyncio
import httpx

AGENT_URL = "http://localhost:8003"


# ── Test patient states ────────────────────────────────────────────────────────

# Patient A: Classic bacterial pneumonia — WBC critical, CRP high
PNEUMONIA_PATIENT = {
    "patient_id": "test-pneumonia-001",
    "demographics": {"name": "John Test", "age": 54, "gender": "male", "dob": "1970-03-14"},
    "active_conditions": [{"code": "J18.9", "display": "Pneumonia", "onset": "2025-04-01"}],
    "medications": [{"drug": "Warfarin", "dose": "5mg", "frequency": "OD", "status": "active"}],
    "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "high"}],
    "lab_results": [
        {"loinc": "26464-8", "display": "White Blood Cell Count", "value": 18.4, "unit": "10*3/uL",
         "reference_high": 11.0, "reference_low": 4.5, "flag": "HIGH"},
        {"loinc": "1988-5", "display": "C-Reactive Protein", "value": 142.0, "unit": "mg/L",
         "reference_high": 10.0, "flag": "HIGH"},
        {"loinc": "718-7", "display": "Hemoglobin", "value": 13.8, "unit": "g/dL",
         "reference_high": 17.5, "reference_low": 13.5, "flag": "NORMAL"},
        {"loinc": "2160-0", "display": "Creatinine", "value": 1.1, "unit": "mg/dL",
         "reference_high": 1.3, "reference_low": 0.7, "flag": "NORMAL"},
        {"loinc": "2823-3", "display": "Potassium", "value": 4.1, "unit": "mEq/L",
         "reference_high": 5.0, "reference_low": 3.5, "flag": "NORMAL"},
    ],
    "diagnostic_reports": [], "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z", "imaging_available": False,
}

# Patient B: Sepsis risk — critical WBC + renal impairment
SEPSIS_PATIENT = {
    "patient_id": "test-sepsis-002",
    "demographics": {"name": "Jane Test", "age": 72, "gender": "female", "dob": "1952-05-20"},
    "active_conditions": [],
    "medications": [],
    "allergies": [],
    "lab_results": [
        {"loinc": "26464-8", "display": "White Blood Cell Count", "value": 31.5, "unit": "10*3/uL",
         "reference_high": 11.0, "flag": "HIGH"},   # Will be classified CRITICAL by rules engine
        {"loinc": "2160-0", "display": "Creatinine", "value": 4.5, "unit": "mg/dL",
         "reference_high": 1.1, "flag": "HIGH"},    # Will be classified CRITICAL
        {"loinc": "2823-3", "display": "Potassium", "value": 6.8, "unit": "mEq/L",
         "reference_high": 5.0, "flag": "HIGH"},    # Will be classified CRITICAL
        {"loinc": "1988-5", "display": "C-Reactive Protein", "value": 250.0, "unit": "mg/L",
         "reference_high": 10.0, "flag": "HIGH"},
    ],
    "diagnostic_reports": [], "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z", "imaging_available": False,
}

# Patient C: Normal labs — nothing flagged
NORMAL_PATIENT = {
    "patient_id": "test-normal-003",
    "demographics": {"name": "Healthy Test", "age": 35, "gender": "female", "dob": "1989-06-10"},
    "active_conditions": [],
    "medications": [],
    "allergies": [],
    "lab_results": [
        {"loinc": "26464-8", "display": "White Blood Cell Count", "value": 7.2, "unit": "10*3/uL",
         "reference_high": 11.0, "reference_low": 4.5, "flag": "NORMAL"},
        {"loinc": "718-7", "display": "Hemoglobin", "value": 13.5, "unit": "g/dL",
         "reference_high": 15.5, "reference_low": 12.0, "flag": "NORMAL"},
        {"loinc": "2160-0", "display": "Creatinine", "value": 0.8, "unit": "mg/dL",
         "reference_high": 1.1, "flag": "NORMAL"},
    ],
    "diagnostic_reports": [], "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z", "imaging_available": False,
}

# Sample diagnosis agent output (for confirmation testing)
DIAGNOSIS_OUTPUT = {
    "top_diagnosis": "Community-acquired pneumonia",
    "top_icd10_code": "J18.9",
    "confidence": 0.87,
}


async def test_lab_analysis_agent():
    async with httpx.AsyncClient(timeout=30.0) as client:

        print("=" * 60)
        print("Testing Lab Analysis Agent — http://localhost:8003")
        print("=" * 60)

        # ── Test 1: Health check ───────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Rules engine: {d.get('rules_engine')}")
            print(f"   LLM ready: {d.get('llm_ready')}")
            print(f"   LOINC codes supported: {d.get('loinc_codes_supported')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy — is it running?")
                return
            print("   ✓ Agent healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start with: python agents/lab_analysis/main.py")
            return

        # ── Test 2: Pneumonia patient — bacterial infection pattern ──
        print("\n2. Pneumonia patient (WBC 18.4 CRITICAL, CRP 142 HIGH)...")
        try:
            r = await client.post(
                f"{AGENT_URL}/analyze-labs",
                json={
                    "patient_state": PNEUMONIA_PATIENT,
                    "diagnosis_agent_output": DIAGNOSIS_OUTPUT,
                },
            )
            d = r.json()

            print(f"   Status: {r.status_code}")
            summary = d["lab_summary"]
            print(f"   Total results: {summary['total_results']}")
            print(f"   Abnormal: {summary['abnormal_count']}")
            print(f"   Critical: {summary['critical_count']}")
            print(f"   Overall severity: {summary['overall_severity']}")

            print(f"\n   Flagged results:")
            for fr in d["flagged_results"]:
                print(f"      {fr['display']}: {fr['value']} {fr['unit']} [{fr['flag']}]")

            patterns = d["pattern_analysis"]["identified_patterns"]
            print(f"\n   Detected patterns ({len(patterns)}):")
            for p in patterns:
                print(f"      - {p['pattern']}")

            print(f"\n   Critical alerts ({len(d['critical_alerts'])}):")
            for a in d["critical_alerts"]:
                print(f"      ⚠️  {a['message'][:80]}...")

            dc = d["diagnosis_confirmation"]
            print(f"\n   Diagnosis confirmation:")
            print(f"      Confirms J18.9: {dc['confirms_top_diagnosis']}")
            print(f"      Confidence boost: +{dc['lab_confidence_boost']}")
            print(f"      Reasoning: {dc['reasoning'][:80]}...")

            # Assertions
            assert summary["overall_severity"] in ("SEVERE", "MODERATE"), \
                f"Expected SEVERE/MODERATE for WBC 18.4 + CRP 142, got {summary['overall_severity']}"
            print("   ✓ Severity correctly SEVERE or MODERATE")

            assert len(d["flagged_results"]) >= 2, "Should flag WBC and CRP"
            print("   ✓ Correct flagged results count")

            assert any("bacterial" in p["pattern"].lower() for p in patterns), \
                "Bacterial infection pattern should be detected"
            print("   ✓ Bacterial infection pattern detected")

            assert len(d["critical_alerts"]) >= 1, "WBC 18.4 should generate at least 1 critical alert"
            print("   ✓ Critical alert generated for WBC 18.4")

            assert "confirms_top_diagnosis" in dc, "diagnosis_confirmation must have confirms_top_diagnosis"
            print("   ✓ Consensus interface field (confirms_top_diagnosis) present")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── Test 3: Sepsis patient — multiple CRITICAL values ────────
        print("\n3. Sepsis patient (WBC 31.5, Creatinine 4.5, K+ 6.8 — all CRITICAL)...")
        try:
            r = await client.post(
                f"{AGENT_URL}/analyze-labs",
                json={"patient_state": SEPSIS_PATIENT},
            )
            d = r.json()

            summary = d["lab_summary"]
            print(f"   Critical count: {summary['critical_count']}")
            print(f"   Severity: {summary['overall_severity']}")
            print(f"   Critical alerts ({len(d['critical_alerts'])}):")
            for a in d["critical_alerts"]:
                print(f"      ⚠️  {a['display']}: {a['value']} — {a['message'][:60]}...")

            assert summary["overall_severity"] == "SEVERE", \
                f"Expected SEVERE for 3+ critical values, got {summary['overall_severity']}"
            print("   ✓ Severity correctly SEVERE")

            assert summary["critical_count"] >= 2, \
                f"Expected ≥2 critical values, got {summary['critical_count']}"
            print(f"   ✓ {summary['critical_count']} critical values correctly flagged")

            alert_loincs = {a["loinc"] for a in d["critical_alerts"]}
            assert "2823-3" in alert_loincs, "Potassium 6.8 must generate critical alert"
            print("   ✓ Critical hyperkalemia alert present (K+ 6.8)")

            assert "2160-0" in alert_loincs, "Creatinine 4.5 must generate critical alert"
            print("   ✓ Critical renal impairment alert present (Cr 4.5)")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── Test 4: Normal patient — no flags, no alerts ─────────────
        print("\n4. Normal patient (all values within range)...")
        try:
            r = await client.post(
                f"{AGENT_URL}/analyze-labs",
                json={"patient_state": NORMAL_PATIENT},
            )
            d = r.json()

            summary = d["lab_summary"]
            print(f"   Abnormal count: {summary['abnormal_count']}")
            print(f"   Critical alerts: {len(d['critical_alerts'])}")
            print(f"   Severity: {summary['overall_severity']}")

            assert summary["abnormal_count"] == 0, \
                f"Normal patient should have 0 abnormal results, got {summary['abnormal_count']}"
            print("   ✓ No abnormal results for normal patient")

            assert len(d["critical_alerts"]) == 0, \
                "Normal patient should have no critical alerts"
            print("   ✓ No critical alerts for normal patient")

            assert summary["overall_severity"] == "NORMAL", \
                f"Expected NORMAL severity, got {summary['overall_severity']}"
            print("   ✓ Severity correctly NORMAL")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── Test 5: Age-adjusted ranges (pediatric) ───────────────────
        print("\n5. Age-adjusted ranges (pediatric patient, WBC 12.0)...")
        try:
            pediatric_state = {
                **NORMAL_PATIENT,
                "patient_id": "test-peds-004",
                "demographics": {"name": "Child Test", "age": 8, "gender": "male", "dob": "2016-01-01"},
                "lab_results": [
                    # WBC 12.0 = NORMAL in pediatric range (5.0-15.0) but HIGH in adult range
                    {"loinc": "26464-8", "display": "WBC", "value": 12.0, "unit": "10*3/uL",
                     "reference_high": 11.0, "flag": "HIGH"},
                ],
            }
            r = await client.post(
                f"{AGENT_URL}/analyze-labs",
                json={"patient_state": pediatric_state},
            )
            d = r.json()
            summary = d["lab_summary"]
            print(f"   Abnormal count for WBC 12.0 in 8-year-old: {summary['abnormal_count']}")
            if summary["abnormal_count"] == 0:
                print("   ✓ WBC 12.0 correctly NORMAL for pediatric patient (range 5.0-15.0)")
            else:
                print("   ⚠️  WBC 12.0 flagged in pediatric patient — check pediatric ranges")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("\n" + "=" * 60)
        print("Lab Analysis Agent tests complete!")
        print("=" * 60)
        print("\nNext step: Build Drug Safety Agent (Agent 4 — MCP Superpower)")


if __name__ == "__main__":
    asyncio.run(test_lab_analysis_agent())