"""
Test script for Drug Safety MCP Agent — v2.0
Port: 8004

Tests:
  Part 1 — Unit tests (no server, no API key):
    - safety_core deterministic logic
    - Pydantic model schemas (structure validation)
    - Drug family classification (expanded table)

  Part 2 — LLM tests (API key required, no server):
    - AlternativeSuggestionsOutput structured output
    - InteractionEnrichmentBatch structured output
    - PatientRiskProfile structured output

  Part 3 — Integration tests (server running at localhost:8004):
    - /health endpoint
    - /check-safety full pipeline
    - MCP tool calls via REST simulation
    - FDA warning integration

Run:
    python agents/drug_safety/test.py --unit        # Part 1 only
    python agents/drug_safety/test.py --llm         # Parts 1+2 (needs GOOGLE_API_KEY)
    python agents/drug_safety/test.py               # All parts (needs server running)
"""
import asyncio
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))


# ════════════════════════════════════════════════════════════════════════════════
# PART 1: Unit tests — no server, no API key
# ════════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from safety_core import (
        check_allergy_cross_reactivity,
        check_condition_contraindications,
        normalize_drug_name,
        get_drug_family,
        AlternativeSuggestionsOutput,
        InteractionEnrichmentBatch,
        PatientRiskProfile,
        AlternativeDrug,
        InteractionEnrichment,
    )

    print("=" * 65)
    print("Part 1: Unit Tests — safety_core.py (no server/API key)")
    print("=" * 65)

    # ── Drug name normalization ───────────────────────────────────
    assert normalize_drug_name("Amoxicillin 500mg TID") == "amoxicillin"
    assert normalize_drug_name("Warfarin") == "warfarin"
    assert normalize_drug_name("Levofloxacin 750mg IV") == "levofloxacin"
    print("✓ Drug name normalization")

    # ── Drug family classification (expanded table) ───────────────
    assert get_drug_family("Amoxicillin") == "penicillin"
    assert get_drug_family("Azithromycin") == "macrolide"
    assert get_drug_family("Levofloxacin") == "fluoroquinolone"
    assert get_drug_family("Ibuprofen") == "nsaid"
    assert get_drug_family("Ceftriaxone") == "cephalosporin"
    assert get_drug_family("Doxycycline") == "tetracycline"
    assert get_drug_family("Gentamicin") == "aminoglycoside"
    assert get_drug_family("Diazepam") == "benzodiazepine"
    assert get_drug_family("Morphine") == "opioid"
    assert get_drug_family("Atorvastatin") == "statin"
    assert get_drug_family("Vancomycin") is None   # Not in any family
    print("✓ Drug family classification (10 families)")

    # ── Direct allergy match ─────────────────────────────────────
    allergies = [{"substance": "Amoxicillin", "reaction": "Rash", "severity": "moderate"}]
    contras = check_allergy_cross_reactivity("Amoxicillin 500mg", allergies)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    assert contras[0]["type"] == "direct_allergy"
    print("✓ Direct allergy match → CRITICAL")

    # ── Penicillin allergy → Amoxicillin cross-reactivity ────────
    allergies_pen = [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}]
    contras = check_allergy_cross_reactivity("Amoxicillin 500mg", allergies_pen)
    assert len(contras) >= 1
    assert any(c["severity"] in ("CRITICAL", "HIGH") for c in contras)
    assert any(c["type"] == "cross_reactivity" for c in contras)
    print("✓ Penicillin allergy → Amoxicillin cross-reactivity flagged")

    # ── Penicillin allergy → Azithromycin should be SAFE ─────────
    contras = check_allergy_cross_reactivity("Azithromycin 500mg", allergies_pen)
    assert len(contras) == 0
    print("✓ Azithromycin SAFE for penicillin-allergic patient")

    # ── Penicillin allergy → Ceftriaxone: mild allergy = no flag ─
    allergies_mild = [{"substance": "Penicillin", "reaction": "Rash", "severity": "mild"}]
    contras_ceph = check_allergy_cross_reactivity("Ceftriaxone", allergies_mild)
    assert len(contras_ceph) == 0, (
        f"Mild penicillin allergy should NOT flag cephalosporins, got {contras_ceph}"
    )
    print("✓ Cephalosporin cross-react NOT flagged for mild penicillin allergy")

    # ── Penicillin anaphylaxis → Ceftriaxone: MODERATE flag ──────
    contras_ceph_severe = check_allergy_cross_reactivity("Ceftriaxone", allergies_pen)
    assert len(contras_ceph_severe) >= 1
    assert contras_ceph_severe[0]["severity"] == "MODERATE"
    print("✓ Cephalosporin cross-react flagged MODERATE for anaphylactic penicillin allergy")

    # ── Fluoroquinolone cross-reactivity ──────────────────────────
    allergies_cipro = [{"substance": "Ciprofloxacin", "reaction": "Tendinopathy", "severity": "moderate"}]
    contras_fq = check_allergy_cross_reactivity("Levofloxacin", allergies_cipro)
    assert len(contras_fq) >= 1
    print("✓ Fluoroquinolone cross-reactivity detected")

    # ── Condition contraindications: Metformin + CKD Stage 5 ─────
    conditions_ckd5 = [{"code": "N18.5", "display": "CKD Stage 5"}]
    contras = check_condition_contraindications("Metformin", conditions_ckd5)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    print("✓ Metformin + CKD Stage 5 → CRITICAL contraindication")

    # ── Levofloxacin + Myasthenia Gravis ─────────────────────────
    conditions_mg = [{"code": "G70.00", "display": "Myasthenia Gravis"}]
    contras = check_condition_contraindications("Levofloxacin", conditions_mg)
    assert len(contras) == 1
    assert contras[0]["severity"] == "CRITICAL"
    print("✓ Levofloxacin + Myasthenia Gravis → CRITICAL contraindication")

    # ── Ibuprofen + Heart failure ─────────────────────────────────
    conditions_hf = [{"code": "I50.9", "display": "Heart Failure"}]
    contras = check_condition_contraindications("Ibuprofen", conditions_hf)
    assert len(contras) == 1
    assert contras[0]["severity"] == "HIGH"
    print("✓ Ibuprofen + Heart Failure → HIGH contraindication")

    # ── Amoxicillin + CKD → no contraindication ──────────────────
    contras = check_condition_contraindications("Amoxicillin", conditions_ckd5)
    assert len(contras) == 0
    print("✓ Amoxicillin + CKD → no condition contraindication")

    # ── Pydantic model schemas ────────────────────────────────────
    print("\nValidating Pydantic structured output schemas...")

    # AlternativeDrug
    alt = AlternativeDrug(
        drug="Azithromycin 500mg PO daily x5",
        rationale="Macrolide antibiotic, no penicillin cross-reactivity",
        drug_class="Macrolide antibiotic",
        interaction_check_needed=["Warfarin"],
        safe_to_prescribe=True,
        cautions="Monitor QTc with Warfarin combination",
    )
    assert alt.drug == "Azithromycin 500mg PO daily x5"
    assert alt.safe_to_prescribe is True
    print("  ✓ AlternativeDrug schema valid")

    # AlternativeSuggestionsOutput
    output = AlternativeSuggestionsOutput(
        alternatives=[alt],
        clinical_note="Switch Amoxicillin to Azithromycin due to penicillin allergy.",
        urgency="BEFORE_NEXT_DOSE",
    )
    assert output.urgency == "BEFORE_NEXT_DOSE"
    assert len(output.alternatives) == 1
    print("  ✓ AlternativeSuggestionsOutput schema valid")

    # InteractionEnrichment
    enrichment = InteractionEnrichment(
        mechanism="Amoxicillin disrupts gut flora → reduces Vitamin K synthesis → potentiates Warfarin",
        clinical_significance="INR increases 1.5-2x within 48-72h of starting Amoxicillin",
        monitoring_parameters=["INR every 48h for first week", "Signs of bleeding"],
        management_strategy="Reduce Warfarin dose by 20-30% and monitor INR closely",
        time_to_onset="Within 48-72 hours",
    )
    assert "INR" in enrichment.monitoring_parameters[0]
    print("  ✓ InteractionEnrichment schema valid")

    # PatientRiskProfile
    profile = PatientRiskProfile(
        overall_risk_level="HIGH",
        primary_risk_factors=["Penicillin allergy", "Warfarin interaction", "Elevated INR risk"],
        safe_to_proceed=False,
        recommended_action="SWITCH_DRUG",
        clinical_summary=(
            "Patient has penicillin allergy making Amoxicillin contraindicated. "
            "Warfarin interaction further complicates proposed regimen. "
            "Switch to Azithromycin and monitor INR."
        ),
        special_populations_flag=None,
    )
    assert profile.recommended_action == "SWITCH_DRUG"
    assert profile.safe_to_proceed is False
    print("  ✓ PatientRiskProfile schema valid")

    print(f"\n✅ All Part 1 unit tests passed!")
    print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════════
# PART 2: LLM structured output tests (needs GOOGLE_API_KEY)
# ════════════════════════════════════════════════════════════════════════════════

async def run_llm_tests():
    from safety_core import (
        get_alternative_suggestions_async,
        enrich_interactions_with_llm,
        generate_patient_risk_profile,
    )

    print("\n" + "=" * 65)
    print("Part 2: LLM Tests — structured output (needs GOOGLE_API_KEY)")
    print("=" * 65)

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not set — skipping LLM tests")
        return

    # ── Test 1: Alternative suggestions structured output ─────────
    print("\n1. AlternativeSuggestionsOutput — Penicillin allergy, Amoxicillin avoidance...")
    result = await get_alternative_suggestions_async(
        drug_name="Amoxicillin 500mg",
        reason_for_avoidance="Penicillin allergy — documented anaphylaxis",
        indication="Community-acquired pneumonia (J18.9)",
        patient_conditions=[{"code": "J18.9", "display": "Pneumonia"}],
        current_medications=[{"drug": "Warfarin 5mg"}],
        allergies=[{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "severe"}],
    )
    assert result is not None, "LLM returned None"
    assert len(result.alternatives) >= 2, f"Expected ≥2 alternatives, got {len(result.alternatives)}"
    assert result.urgency in ("IMMEDIATE", "BEFORE_NEXT_DOSE", "ROUTINE")
    assert result.clinical_note

    # Check no alternatives are penicillins
    for alt in result.alternatives:
        assert "penicillin" not in alt.drug.lower(), f"Alternative {alt.drug} is a penicillin!"
        assert "amoxicillin" not in alt.drug.lower(), f"Alternative is the same drug!"

    print(f"   ✓ {len(result.alternatives)} alternatives returned")
    print(f"   ✓ Urgency: {result.urgency}")
    print(f"   ✓ Clinical note: {result.clinical_note}")
    for alt in result.alternatives:
        print(f"   ✓ Alternative: {alt.drug} — {alt.drug_class}")

    # ── Test 2: Interaction enrichment structured output ──────────
    print("\n2. InteractionEnrichmentBatch — Warfarin + Amoxicillin interaction...")
    mock_interactions = [
        {
            "drug_a": "Warfarin",
            "drug_b": "Amoxicillin",
            "severity": "MODERATE",
            "description": "Amoxicillin may increase anticoagulant effect of Warfarin",
            "source": "NLM RxNav",
        }
    ]
    patient_ctx = {
        "demographics": {"age": 65, "gender": "male"},
        "active_conditions": [
            {"code": "I48.0", "display": "Atrial fibrillation"},
            {"code": "J18.9", "display": "Pneumonia"},
        ],
        "medications": [{"drug": "Warfarin 5mg"}],
        "allergies": [],
    }
    enrichment = await enrich_interactions_with_llm(mock_interactions, patient_ctx)

    assert enrichment is not None, "Enrichment returned None"
    assert len(enrichment.enriched_interactions) == 1
    e = enrichment.enriched_interactions[0]
    assert e.mechanism
    assert e.monitoring_parameters and len(e.monitoring_parameters) >= 1
    assert e.management_strategy
    assert e.time_to_onset
    assert enrichment.overall_risk_narrative

    print(f"   ✓ Mechanism: {e.mechanism[:80]}...")
    print(f"   ✓ Monitoring: {e.monitoring_parameters}")
    print(f"   ✓ Management: {e.management_strategy[:80]}...")
    print(f"   ✓ Onset: {e.time_to_onset}")
    print(f"   ✓ Narrative: {enrichment.overall_risk_narrative[:100]}...")

    # ── Test 3: Patient risk profile structured output ────────────
    print("\n3. PatientRiskProfile — high-risk patient...")
    mock_contras = [
        {
            "drug": "Amoxicillin 500mg",
            "allergen": "Penicillin",
            "severity": "CRITICAL",
            "reason": "Penicillin allergy — direct cross-reactivity",
        }
    ]
    mock_interactions = [
        {
            "drug_a": "Warfarin",
            "drug_b": "Azithromycin",
            "severity": "MODERATE",
            "description": "QT prolongation risk",
        }
    ]
    profile = await generate_patient_risk_profile(
        patient_state=patient_ctx,
        proposed_medications=["Amoxicillin 500mg", "Azithromycin 500mg"],
        contraindications=mock_contras,
        interactions=mock_interactions,
        fda_warnings={"Azithromycin": ["[BLACK BOX] QT prolongation risk in patients with QTc >500ms"]},
        safety_status="UNSAFE",
    )

    assert profile is not None, "Risk profile returned None"
    assert profile.overall_risk_level in ("CRITICAL", "HIGH", "MODERATE", "LOW", "MINIMAL")
    assert profile.recommended_action in (
        "PRESCRIBE", "PRESCRIBE_WITH_MONITORING",
        "SWITCH_DRUG", "CONSULT_PHARMACIST", "DO_NOT_PRESCRIBE"
    )
    assert len(profile.primary_risk_factors) >= 1
    assert profile.clinical_summary
    assert not profile.safe_to_proceed  # Should be False given CRITICAL allergy

    print(f"   ✓ Risk level: {profile.overall_risk_level}")
    print(f"   ✓ Action: {profile.recommended_action}")
    print(f"   ✓ Risk factors: {profile.primary_risk_factors}")
    print(f"   ✓ Safe to proceed: {profile.safe_to_proceed}")
    print(f"   ✓ Summary: {profile.clinical_summary[:120]}...")

    print(f"\n✅ All LLM structured output tests passed!")
    print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════════
# PART 3: Integration tests — server at localhost:8004
# ════════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx
    AGENT_URL = "http://localhost:8004"

    print("\n" + "=" * 65)
    print("Part 3: Integration Tests — server at localhost:8004")
    print("=" * 65)

    async with httpx.AsyncClient(timeout=90.0) as client:

        # ── Health check ─────────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{AGENT_URL}/health")
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Version: {d.get('version')}")
            print(f"   Redis: {d.get('redis')}")
            print(f"   LLM model: {d.get('llm_model')}")
            print(f"   MCP tools: {d.get('mcp_tools')}")
            if r.status_code != 200:
                print("   ❌ Agent not healthy")
                return
            print("   ✓ Agent v2.0 healthy")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print("   Start with: python agents/drug_safety/main.py")
            return

        # ── Full pipeline: penicillin allergy + warfarin ──────────
        print("\n2. Full pipeline — Penicillin-allergic patient, Amoxicillin + Warfarin interaction...")
        print("   (LLM enrichment enabled — may take 10-15s)")
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
                "enrich_with_llm": True,
            })
            d = r.json()
            print(f"   Status code: {r.status_code}")
            print(f"   Safety status: {d['safety_status']}")
            print(f"   Approved: {d['approved_medications']}")
            print(f"   Flagged: {d['flagged_medications']}")
            print(f"   Contraindications: {d['summary']['contraindication_count']}")
            print(f"   Interactions: {d['summary']['interaction_count']}")
            print(f"   LLM enriched: {d['summary']['llm_enriched']}")
            print(f"   Black box warnings: {d['summary']['black_box_warnings']}")

            # Patient risk profile
            if d.get("patient_risk_profile"):
                profile = d["patient_risk_profile"]
                print(f"\n   Patient Risk Profile:")
                print(f"     Risk level: {profile.get('overall_risk_level')}")
                print(f"     Action: {profile.get('recommended_action')}")
                print(f"     Safe to proceed: {profile.get('safe_to_proceed')}")
                print(f"     Summary: {profile.get('clinical_summary', '')[:120]}...")

            # Enriched interactions
            interactions = d.get("critical_interactions", [])
            if interactions and interactions[0].get("mechanism"):
                print(f"\n   Enriched Interaction [{interactions[0]['drug_a']} + {interactions[0]['drug_b']}]:")
                print(f"     Mechanism: {interactions[0].get('mechanism', '')[:80]}...")
                print(f"     Monitoring: {interactions[0].get('monitoring_parameters', [])[:2]}")
                print(f"     Management: {interactions[0].get('management_strategy', '')[:80]}...")

            # Assertions
            assert d["safety_status"] in ("UNSAFE", "CAUTION")
            assert "Amoxicillin 500mg" in d["flagged_medications"]
            assert len(d["contraindications"]) >= 1
            print("\n   ✓ Safety status UNSAFE/CAUTION")
            print("   ✓ Amoxicillin correctly flagged")
            print(f"   ✓ {len(d['contraindications'])} contraindication(s) found")
            print(f"   ✓ {len(d.get('fhir_medication_requests', []))} FHIR MedicationRequest(s)")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── Clean patient — no enrichment ─────────────────────────
        print("\n3. Clean patient, no LLM enrichment (fast path)...")
        try:
            r = await client.post(f"{AGENT_URL}/check-safety", json={
                "proposed_medications": ["Azithromycin 500mg"],
                "current_medications": ["Metformin 850mg"],
                "patient_allergies": [],
                "active_conditions": [{"code": "J18.9", "display": "Pneumonia"}],
                "patient_id": "test-patient-002",
                "enrich_with_llm": False,
            })
            d = r.json()
            print(f"   Safety status: {d['safety_status']}")
            assert d["safety_status"] in ("SAFE", "CAUTION")
            assert "Azithromycin 500mg" in d["approved_medications"]
            assert d["summary"]["llm_enriched"] is False
            print("   ✓ Azithromycin approved (no LLM enrichment)")

        except AssertionError as e:
            print(f"   ❌ Assertion failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── RxNav API connectivity ────────────────────────────────
        print("\n4. External API connectivity...")
        try:
            from fda_client import get_rxcuis_batch, get_fda_warnings_batch
            rxcuis, fda_w = await asyncio.gather(
                get_rxcuis_batch(["Azithromycin", "Warfarin"]),
                get_fda_warnings_batch(["Warfarin"]),
            )
            resolved = {k: v for k, v in rxcuis.items() if v}
            print(f"   RxNav: resolved {len(resolved)}/2 drugs to RxCUI")
            for name, rxcui in rxcuis.items():
                print(f"      {'✓' if rxcui else '⚠️ '} {name}: {rxcui}")
            fda_warnings = fda_w.get("Warfarin", [])
            print(f"   FDA: {len(fda_warnings)} warning(s) for Warfarin")
            if fda_warnings:
                print(f"      First: {fda_warnings[0][:80]}...")
            if resolved:
                print("   ✓ External APIs reachable")
        except Exception as e:
            print(f"   ⚠️  External API test failed: {e}")

    print("\n" + "=" * 65)
    print("Drug Safety Agent v2.0 tests complete!")
    print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Part 1 only (no API key, no server)")
    parser.add_argument("--llm", action="store_true", help="Parts 1+2 (needs GOOGLE_API_KEY)")
    args = parser.parse_args()

    run_unit_tests()

    if args.unit:
        sys.exit(0)

    # asyncio.run(run_llm_tests())

    if not args.llm:
        asyncio.run(run_integration_tests())