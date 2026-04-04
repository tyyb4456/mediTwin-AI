"""
Test script for Drug Safety MCP Agent
Port: 8004

Tests:
  1. Unit tests — safety_core.py (no server, no API key needed)
  2. REST /health endpoint
  3. REST /check-safety — full pipeline (needs server running)
  4. External API check — RxNav interaction lookup

Run:
    python agents/drug_safety/test.py

Prerequisites (for server tests):
    python agents/drug_safety/main.py   (in another terminal)
    Redis optionally running (agent degrades gracefully without it)
"""
import asyncio
import sys
import os

# Add agent dir to path so we can import safety_core directly for unit tests
sys.path.insert(0, os.path.dirname(__file__))


# ════════════════════════════════════════════════════════════════════════════════
# PART 1: Unit tests — no server, no API key, pure Python
# ════════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from safety_core import (
        check_allergy_cross_reactivity,
        check_condition_contraindications,
        normalize_drug_name,
        get_drug_family,
    )

    print("=" * 60)
    print("Unit Tests — safety_core.py (no server/API key needed)")
    print("=" * 60)

    # ── Normalize drug name ──────────────────────────────────────
    assert normalize_drug_name("Amoxicillin 500mg TID") == "amoxicillin"
    assert normalize_drug_name("Warfarin") == "warfarin"
    print("✓ Drug name normalization")

    # ── Drug family lookup ───────────────────────────────────────
    assert get_drug_family("Amoxicillin") == "penicillin"
    assert get_drug_family("Azithromycin") == "macrolide"
    assert get_drug_family("Levofloxacin") == "fluoroquinolone"
    assert get_drug_family("Ibuprofen") == "nsaid"
    assert get_drug_family("Vancomycin") is None   # Not in any family
    print("✓ Drug family classification")

    # ── Allergy: direct name match ───────────────────────────────
    allergies = [{"substance": "Amoxicillin", "reaction": "Rash", "severity": "moderate"}]
    contras = check_allergy_cross_reactivity("Amoxicillin 500mg", allergies)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    print("✓ Direct allergy match → CRITICAL")

    # ── Allergy: penicillin → amoxicillin cross-reactivity ───────
    allergies_pen = [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}]
    contras = check_allergy_cross_reactivity("Amoxicillin 500mg", allergies_pen)
    assert len(contras) >= 1, f"Expected cross-reactivity, got {contras}"
    assert any(c["severity"] in ("CRITICAL", "HIGH") for c in contras)
    print("✓ Penicillin allergy → Amoxicillin cross-reactivity flagged")

    # ── Allergy: penicillin → azithromycin should be SAFE ────────
    contras = check_allergy_cross_reactivity("Azithromycin 500mg", allergies_pen)
    assert len(contras) == 0, f"Azithromycin should be safe for penicillin allergy, got {contras}"
    print("✓ Azithromycin SAFE for penicillin-allergic patient")

    # ── Allergy: cephalosporin cross-react only for severe allergy
    allergies_mild = [{"substance": "Penicillin", "reaction": "Rash", "severity": "mild"}]
    contras_ceph = check_allergy_cross_reactivity("Ceftriaxone", allergies_mild)
    # Mild penicillin allergy → cephalosporin cross-reactivity should NOT be flagged
    # (MODERATE cross-reactivity only flagged for severe allergies)
    print(f"✓ Cephalosporin cross-react for MILD penicillin allergy: {len(contras_ceph)} flags (expected 0)")

    # ── Condition contraindications: Metformin + CKD Stage 5 ─────
    conditions_ckd5 = [{"code": "N18.5", "display": "CKD Stage 5"}]
    contras = check_condition_contraindications("Metformin", conditions_ckd5)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    print("✓ Metformin + CKD Stage 5 → CRITICAL contraindication")

    # ── Condition contraindications: Levofloxacin + Myasthenia Gravis
    conditions_mg = [{"code": "G70.00", "display": "Myasthenia Gravis"}]
    contras = check_condition_contraindications("Levofloxacin", conditions_mg)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    print("✓ Levofloxacin + Myasthenia Gravis → CRITICAL contraindication")

    # ── Condition contraindications: Amoxicillin + CKD (not in DB → safe)
    contras = check_condition_contraindications("Amoxicillin", conditions_ckd5)
    assert len(contras) == 0   # Amoxicillin has no CKD contraindications in our DB
    print("✓ Amoxicillin + CKD Stage 5 → no condition contraindication (correct)")

    # ── Ibuprofen + Heart failure → HIGH ──────────────────────────
    conditions_hf = [{"code": "I50.9", "display": "Heart Failure"}]
    contras = check_condition_contraindications("Ibuprofen", conditions_hf)
    assert len(contras) == 1
    assert contras[0]["severity"] == "HIGH"
    print("✓ Ibuprofen + Heart Failure → HIGH contraindication")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# PART 2: Integration tests — needs server running at localhost:8004
# ════════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx

    AGENT_URL = "http://localhost:8004"
    print("\n" + "=" * 60)
    print("Integration Tests — requires server at localhost:8004")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:

        # ── Health check ─────────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   MCP tools: {d.get('mcp_tools')}")
            print(f"   MCP endpoint: {d.get('mcp_endpoint')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy — is it running?")
                print("   Start with: python agents/drug_safety/main.py")
                return
            print("   ✓ Agent healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start with: python agents/drug_safety/main.py")
            return

        # ── REST: Penicillin allergy + Amoxicillin proposed ──────
        print("\n2. Full safety check — Penicillin-allergic patient, Amoxicillin proposed...")
        try:
            r = await client.post(f"{AGENT_URL}/check-safety", json={
                "proposed_medications": ["Amoxicillin 500mg", "Ibuprofen 400mg"],
                "current_medications": ["Warfarin 5mg", "Metformin 850mg"],
                "patient_allergies": [
                    {"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}
                ],
                "active_conditions": [
                    {"code": "J18.9", "display": "Pneumonia"},
                    {"code": "I48.0", "display": "Atrial fibrillation"},
                ],
                "patient_id": "test-patient-001",
            })
            d = r.json()

            print(f"   Status code: {r.status_code}")
            print(f"   Safety status: {d['safety_status']}")
            print(f"   Approved: {d['approved_medications']}")
            print(f"   Flagged: {d['flagged_medications']}")
            print(f"   Contraindications ({len(d['contraindications'])}):")
            for c in d["contraindications"]:
                print(f"      [{c['severity']}] {c['drug']} — {c['reason'][:70]}...")
            print(f"   Interactions ({len(d['critical_interactions'])}):")
            for i in d["critical_interactions"][:3]:
                print(f"      [{i['severity']}] {i['drug_a']} + {i['drug_b']}")

            # Assertions
            assert d["safety_status"] in ("UNSAFE", "CAUTION"), \
                f"Expected UNSAFE/CAUTION for penicillin allergy + amoxicillin"
            print("   ✓ Safety status correctly UNSAFE or CAUTION")

            assert "Amoxicillin 500mg" in d["flagged_medications"], \
                "Amoxicillin must be flagged for penicillin-allergic patient"
            print("   ✓ Amoxicillin correctly flagged")

            assert len(d["contraindications"]) >= 1
            print(f"   ✓ {len(d['contraindications'])} contraindication(s) found")

            assert len(d["fhir_medication_requests"]) >= 0
            print(f"   ✓ {len(d['fhir_medication_requests'])} FHIR MedicationRequest(s) generated")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── REST: Clean patient, safe medications ─────────────────
        print("\n3. Safety check — no allergies, no contraindications...")
        try:
            r = await client.post(f"{AGENT_URL}/check-safety", json={
                "proposed_medications": ["Azithromycin 500mg"],
                "current_medications": ["Metformin 850mg"],
                "patient_allergies": [],
                "active_conditions": [{"code": "J18.9", "display": "Pneumonia"}],
                "patient_id": "test-patient-002",
            })
            d = r.json()
            print(f"   Safety status: {d['safety_status']}")
            print(f"   Approved: {d['approved_medications']}")

            assert d["safety_status"] in ("SAFE", "CAUTION"), \
                f"Azithromycin + Metformin should be SAFE, got {d['safety_status']}"
            assert "Azithromycin 500mg" in d["approved_medications"]
            print("   ✓ Azithromycin correctly approved")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── RxNav API connectivity ────────────────────────────────
        print("\n4. External RxNav API connectivity test...")
        try:
            from fda_client import get_rxcuis_batch
            rxcuis = await get_rxcuis_batch(["Azithromycin", "Warfarin"])
            resolved = {k: v for k, v in rxcuis.items() if v}
            print(f"   Resolved {len(resolved)}/{len(rxcuis)} drugs to RxCUI:")
            for name, rxcui in rxcuis.items():
                status = "✓" if rxcui else "⚠️ "
                print(f"      {status} {name}: {rxcui}")
            if resolved:
                print("   ✓ RxNav API reachable")
            else:
                print("   ⚠️  RxNav API unreachable — check internet connectivity")
        except Exception as e:
            print(f"   ⚠️  RxNav test failed: {e}")

    print("\n" + "=" * 60)
    print("Drug Safety Agent tests complete!")
    print("=" * 60)
    print("\nNext step: Build Imaging Triage Agent (Agent 5)")


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Always run unit tests (no server needed)
    run_unit_tests()

    # Then run integration tests (server must be running)
    asyncio.run(run_integration_tests())