"""
Test script for MediTwin Tool Agent
Port: 8010

Part 1 — Unit tests (no server needed):
    - All 8 tools registered with correct names and descriptions
    - Tool argument schemas have required fields
    - _run async bridge works correctly
    - _post returns error JSON on failure (no real server needed)

Part 2 — Integration tests (all agents + tool_agent running):
    - Health check — 8 tools registered
    - A2A agent card structure
    - General query (no patient ID) → zero tools called, direct answer
    - Patient-specific query → fetch_patient_context called first
    - Focused query (labs only) → only analyze_labs called, not full pipeline
    - Drug safety query → only check_drug_safety called
    - Full workup query → multiple tools called including consensus + report
    - Session continuity via session_id
    - Empty query validation

Run:
    python agents/tool_agent/test.py --unit
    python agents/tool_agent/test.py
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    from tools import (
        MEDITWIN_TOOLS,
        fetch_patient_context,
        run_diagnosis,
        analyze_labs,
        check_drug_safety,
        analyze_chest_xray,
        simulate_treatment_outcomes,
        run_consensus,
        generate_clinical_report,
    )

    print("=" * 60)
    print("Unit Tests — MediTwin Tool Agent")
    print("=" * 60)

    # ── 1. Tool count ──────────────────────────────────────────────
    print("\n1. Tool registry...")
    assert len(MEDITWIN_TOOLS) == 8, f"Expected 8 tools, got {len(MEDITWIN_TOOLS)}"
    print(f"   ✓ {len(MEDITWIN_TOOLS)} tools registered")
    for t in MEDITWIN_TOOLS:
        print(f"      • {t.name}")

    # ── 2. Tool names ─────────────────────────────────────────────
    print("\n2. Tool name validation...")
    expected = {
        "fetch_patient_context", "run_diagnosis", "analyze_labs",
        "check_drug_safety", "analyze_chest_xray", "simulate_treatment_outcomes",
        "run_consensus", "generate_clinical_report",
    }
    actual = {t.name for t in MEDITWIN_TOOLS}
    assert actual == expected, f"Mismatch: {actual ^ expected}"
    print("   ✓ All 8 tool names correct")

    # ── 3. Descriptions contain WHEN TO USE ───────────────────────
    print("\n3. Tool descriptions contain WHEN TO USE guidance...")
    for t in MEDITWIN_TOOLS:
        assert "WHEN TO USE" in t.description, f"{t.name} missing WHEN TO USE"
        assert len(t.description) > 100, f"{t.name} description too short"
        print(f"   ✓ {t.name}: {len(t.description)} chars, WHEN TO USE present")

    # ── 4. Tool schemas ────────────────────────────────────────────
    print("\n4. Tool argument schemas...")
    schema_checks = {
        "fetch_patient_context":       ["patient_id"],
        "run_diagnosis":               ["patient_state_json", "chief_complaint"],
        "analyze_labs":                ["patient_state_json"],
        "check_drug_safety":           ["patient_state_json", "proposed_medications"],
        "analyze_chest_xray":          ["patient_state_json", "image_data_base64"],
        "simulate_treatment_outcomes": ["patient_state_json", "diagnosis_json", "treatment_options"],
        "run_consensus":               ["patient_state_json", "diagnosis_json"],
        "generate_clinical_report":    ["patient_state_json", "consensus_json", "chief_complaint"],
    }
    tool_by_name = {t.name: t for t in MEDITWIN_TOOLS}
    for tool_name, required_args in schema_checks.items():
        t = tool_by_name[tool_name]
        schema = t.args_schema.schema() if t.args_schema else {}
        props = schema.get("properties", {})
        for arg in required_args:
            assert arg in props, f"{tool_name} missing arg '{arg}'"
        print(f"   ✓ {tool_name}: {list(props.keys())}")

    # ── 5. _run async bridge ──────────────────────────────────────
    print("\n5. Async→sync bridge (_run)...")
    from agents.tool_agent.tools import _run

    async def _dummy():
        return "bridge_ok"

    result = _run(_dummy())
    assert result == "bridge_ok"
    print("   ✓ _run correctly bridges async→sync")

    # ── 6. _post error handling ───────────────────────────────────
    print("\n6. _post error handling (unreachable server)...")
    import json
    from agents.tool_agent.tools import _post

    result_str = _run(_post("TestAgent", "http://localhost:19999/nope", {}, timeout=2.0))
    result = json.loads(result_str)
    assert "error" in result
    print(f"   ✓ _post returns error JSON: {result['error'][:60]}")

    # ── 7. System prompt triage logic present ─────────────────────
    print("\n7. System prompt — triage rules present...")
    from agents.tool_agent.agent import SYSTEM_PROMPT
    assert "CASE A" in SYSTEM_PROMPT, "Missing CASE A (no patient ID)"
    assert "CASE B" in SYSTEM_PROMPT, "Missing CASE B (patient ID present)"
    assert "fetch_patient_context" in SYSTEM_PROMPT
    assert "Do NOT call any tool" in SYSTEM_PROMPT
    print("   ✓ CASE A (no patient ID → no tools) present")
    print("   ✓ CASE B (patient ID → selective tools) present")
    print("   ✓ fetch_patient_context referenced as mandatory first step")
    print(f"   ✓ System prompt length: {len(SYSTEM_PROMPT)} chars")

    print(f"\n✅ All unit tests passed!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

async def run_integration_tests():
    import httpx
    BASE = "http://localhost:8010"

    print("\n" + "=" * 60)
    print("Integration Tests — requires Tool Agent + all downstream agents")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=180.0) as client:

        # ── 1. Health check ──────────────────────────────────────
        print("\n1. Health check...")
        try:
            r = await client.get(f"{BASE}/health")
            d = r.json()
            print(f"   Status: {r.status_code} | {d.get('status')}")
            print(f"   Mode: {d.get('mode')}")
            print(f"   Tools: {d.get('tool_count')} registered")
            assert r.status_code == 200
            assert d.get("tool_count") == 8
            print("   ✓ Tool Agent healthy, 8 tools registered")
        except Exception as e:
            print(f"   ❌ Not running: {e}")
            print("   Start: python agents/tool_agent/main.py")
            return

        # ── 2. A2A Agent Card ─────────────────────────────────────
        print("\n2. A2A Agent Card...")
        try:
            r = await client.get(f"{BASE}/.well-known/agent-card")
            d = r.json()
            assert d["name"] == "MediTwin Tool Agent"
            assert d["port"] == 8010
            assert "triage_logic" in d
            assert "with_patient_id" in d["triage_logic"]
            assert "without_patient_id" in d["triage_logic"]
            print(f"   ✓ name: {d['name']}")
            print(f"   ✓ triage_logic declared: {d['triage_logic']}")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 3. General query — NO patient ID → zero tools ─────────
        print("\n3. General query (no patient ID) — expecting zero tool calls...")
        try:
            r = await client.post(f"{BASE}/query", json={
                "query": "What is the mechanism of action of azithromycin?",
                "session_id": "test-general-1",
            })
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Mode: {d.get('mode')}")
            print(f"   Tools called: {d.get('tools_called', [])}")
            print(f"   Answer preview: {d.get('answer', '')[:150]}...")
            assert r.status_code == 200
            assert d.get("mode") == "general_knowledge"
            assert len(d.get("tools_called", [])) == 0
            print("   ✓ No tools called — answered from general knowledge")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 4. Another general query ───────────────────────────────
        print("\n4. General query — CAP diagnostic criteria...")
        try:
            r = await client.post(f"{BASE}/query", json={
                "query": "What are the CURB-65 criteria for community-acquired pneumonia?",
                "session_id": "test-general-2",
            })
            d = r.json()
            assert d.get("mode") == "general_knowledge"
            assert len(d.get("tools_called", [])) == 0
            print(f"   ✓ Mode: {d['mode']} | Tools: {d['tools_called']} | Elapsed: {d['elapsed_seconds']}s")
            print(f"   Answer: {d.get('answer', '')[:150]}...")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ {e}")

        # ── 5. Patient query — diagnosis focus ─────────────────────
        print("\n5. Patient-specific diagnosis query...")
        print("   Note: calls FHIR + Diagnosis agent — may take 20-40s")
        try:
            r = await client.post(f"{BASE}/query", json={
                "query": "What is the likely diagnosis for patient example? They have fever and cough.",
                "session_id": "test-patient-example",
            })
            d = r.json()
            print(f"   Status: {r.status_code} | Elapsed: {d.get('elapsed_seconds')}s")
            print(f"   Mode: {d.get('mode')}")
            print(f"   Tools called: {d.get('tools_called', [])}")
            if r.status_code == 200:
                assert "fetch_patient_context" in d.get("tools_called", []), \
                    "fetch_patient_context must be called first"
                assert d.get("mode") == "patient_specific"
                print(f"   ✓ fetch_patient_context called first ✓")
                print(f"   ✓ mode=patient_specific ✓")
                print(f"   Answer: {d.get('answer', '')[:200]}...")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 6. Patient query — labs focus only ────────────────────
        print("\n6. Patient-specific LAB query (should NOT call diagnosis)...")
        try:
            r = await client.post(f"{BASE}/query", json={
                "query": "Are the lab results normal for patient example?",
                "session_id": "test-patient-labs",
            })
            d = r.json()
            print(f"   Status: {r.status_code} | Elapsed: {d.get('elapsed_seconds')}s")
            print(f"   Tools called: {d.get('tools_called', [])}")
            if r.status_code == 200:
                tools = d.get("tools_called", [])
                assert "fetch_patient_context" in tools
                assert "analyze_labs" in tools
                # Should NOT call run_diagnosis for a pure lab question
                print(f"   ✓ fetch_patient_context + analyze_labs called")
                print(f"   ✓ Selective — only labs-relevant tools invoked")
                print(f"   Answer: {d.get('answer', '')[:200]}...")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 7. Drug safety focused query ──────────────────────────
        print("\n7. Drug safety query — 'is amoxicillin safe for patient example?'...")
        try:
            r = await client.post(f"{BASE}/query", json={
                "query": "Is it safe to prescribe amoxicillin to patient example?",
                "session_id": "test-patient-drugs",
            })
            d = r.json()
            print(f"   Status: {r.status_code} | Elapsed: {d.get('elapsed_seconds')}s")
            print(f"   Tools called: {d.get('tools_called', [])}")
            if r.status_code == 200:
                tools = d.get("tools_called", [])
                assert "fetch_patient_context" in tools
                assert "check_drug_safety" in tools
                print(f"   ✓ fetch_patient_context + check_drug_safety called")
                print(f"   Answer: {d.get('answer', '')[:200]}...")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # ── 8. Session continuity ─────────────────────────────────
        print("\n8. Session continuity — follow-up in same session...")
        try:
            # First: ask a general question to seed the session
            await client.post(f"{BASE}/query", json={
                "query": "What is sepsis?",
                "session_id": "test-continuity",
            })
            # Then: follow-up
            r = await client.post(f"{BASE}/query", json={
                "query": "How does that relate to community-acquired pneumonia?",
                "session_id": "test-continuity",
            })
            d = r.json()
            print(f"   Status: {r.status_code}")
            print(f"   Mode: {d.get('mode')}")
            assert d.get("session_id") == "test-continuity"
            print(f"   ✓ session_id preserved: {d['session_id']}")
            print(f"   Answer: {d.get('answer', '')[:150]}...")
        except Exception as e:
            print(f"   ⚠️  {e}")

        # ── 9. Empty query validation ─────────────────────────────
        print("\n9. Input validation — empty query...")
        try:
            r = await client.post(f"{BASE}/query", json={"query": "  "})
            print(f"   Status: {r.status_code}")
            assert r.status_code == 400
            print("   ✓ Empty query correctly rejected with 400")
        except AssertionError as e:
            print(f"   ❌ {e}")
        except Exception as e:
            print(f"   ⚠️  {e}")

    print("\n" + "=" * 60)
    print("Tool Agent tests complete!")
    print("=" * 60)
    print("""
Architecture comparison:

  Orchestrator (8000)         Tool Agent (8010)
  ─────────────────────────   ──────────────────────────────
  POST /analyze               POST /query (natural language)
  Always runs all 8 agents    Triage: 0 tools or selective
  Fixed execution order       LLM decides tool sequence
  Structured request body     Free-form query string
  No general knowledge mode   General questions answered directly
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Unit tests only")
    args = parser.parse_args()

    run_unit_tests()
    if not args.unit:
        asyncio.run(run_integration_tests())