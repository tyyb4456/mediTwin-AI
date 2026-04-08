"""
Test script for Diagnosis Agent
Tests against locally running agent at http://localhost:8002

Prerequisites:
    1. ChromaDB running: docker run -p 8008:8000 chromadb/chroma:latest
    2. Knowledge base seeded: python knowledge_base/ingest.py
    3. Agent running: python agents/diagnosis/main.py
    4. OPENAI_API_KEY set in environment

Usage:
    python agents/diagnosis/test.py
"""
import asyncio
import httpx
import json


# Sample patient state — mimics Patient Context Agent output for a pneumonia patient
SAMPLE_PATIENT_STATE = {
    "patient_id": "test-patient-001",
    "demographics": {
        "name": "John Test",
        "age": 54,
        "gender": "male",
        "dob": "1970-03-14"
    },
    "active_conditions": [
        {"code": "E11.9", "display": "Type 2 diabetes mellitus", "onset": "2018-01-01"},
        {"code": "I48.0", "display": "Atrial fibrillation", "onset": "2020-06-15"}
    ],
    "medications": [
        {"drug": "Warfarin", "dose": "5mg", "frequency": "OD", "status": "active"},
        {"drug": "Metformin", "dose": "850mg", "frequency": "BID", "status": "active"}
    ],
    "allergies": [
        {"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "high"}
    ],
    "lab_results": [
        {
            "loinc": "26464-8",
            "display": "White Blood Cell Count",
            "value": 18.4,
            "unit": "10*3/uL",
            "reference_high": 11.0,
            "reference_low": 4.5,
            "flag": "CRITICAL"
        },
        {
            "loinc": "718-7",
            "display": "Hemoglobin",
            "value": 13.8,
            "unit": "g/dL",
            "reference_high": 17.5,
            "reference_low": 13.5,
            "flag": "NORMAL"
        },
        {
            "loinc": "1988-5",
            "display": "C-Reactive Protein",
            "value": 142.0,
            "unit": "mg/L",
            "reference_high": 10.0,
            "flag": "HIGH"
        }
    ],
    "diagnostic_reports": [],
    "recent_encounters": [],
    "state_timestamp": "2025-04-01T10:30:00Z",
    "imaging_available": False
}

AGENT_URL = "http://localhost:8002"


async def test_diagnosis_agent():
    """Run all tests against the Diagnosis Agent"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 60)
        print("Testing Diagnosis Agent — http://localhost:8002")
        print("=" * 60)
        
        # ── Test 1: Health Check ──────────────────────────────────
        print("\n1. Health check...")
        try:
            response = await client.get(f"{AGENT_URL}/health")
            data = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Agent: {data.get('agent')}")
            print(f"   RAG ready: {data.get('rag_ready')}")
            
            if not data.get("rag_ready"):
                print("\n   ⚠️  RAG is not ready!")
                print("   Steps to fix:")
                print("   1. Start ChromaDB: docker run -d -p 8008:8000 chromadb/chroma:latest")
                print("   2. Seed KB: python knowledge_base/ingest.py")
                print("   3. Restart this agent")
                return
            else:
                print("   ✓ RAG is ready")
        except Exception as e:
            print(f"   ❌ Agent not running: {e}")
            print(f"   Start it with: python agents/diagnosis/main.py")
            return
        
        # ── Test 2: Diagnosis Endpoint ────────────────────────────
        print("\n2. Testing /diagnose endpoint...")
        print("   (This calls Gemini — takes 5-15 seconds)")
        try:
            response = await client.post(
                f"{AGENT_URL}/diagnose",
                json={
                    "patient_state": SAMPLE_PATIENT_STATE,
                    "chief_complaint": "Productive cough, fever 38.9°C, shortness of breath for 3 days",
                    "include_fhir_resources": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n   ✓ Diagnosis successful!")
                print(f"\n   Top Diagnosis: {data['top_diagnosis']}")
                print(f"   ICD-10: {data['top_icd10_code']}")
                print(f"   Confidence: {data['confidence_level']}")
                print(f"   RAG used: {data['rag_available']}")
                
                print(f"\n   Differential ({len(data['differential_diagnosis'])} diagnoses):")
                for diag in data["differential_diagnosis"]:
                    print(f"      {diag['rank']}. {diag['display']} ({diag['icd10_code']}) — {diag['confidence']:.0%}")
                
                print(f"\n   Reasoning: {data['reasoning_summary'][:200]}...")
                
                print(f"\n   Recommended Next Steps:")
                for step in data["recommended_next_steps"][:5]:
                    print(f"      - {step}")
                
                # Validate structure
                print("\n   Structure validation:")
                assert data["top_icd10_code"], "top_icd10_code must be set"
                print("   ✓ top_icd10_code present")
                
                assert len(data["differential_diagnosis"]) >= 1, "Must have at least 1 diagnosis"
                print(f"   ✓ {len(data['differential_diagnosis'])} diagnoses in differential")
                
                top = data["differential_diagnosis"][0]
                assert "icd10_code" in top, "Each diagnosis must have icd10_code"
                assert "confidence" in top, "Each diagnosis must have confidence"
                assert "supporting_evidence" in top, "Each diagnosis must have supporting_evidence"
                print("   ✓ Diagnosis structure valid (icd10_code, confidence, evidence)")
                
                if data.get("fhir_conditions"):
                    print(f"   ✓ {len(data['fhir_conditions'])} FHIR Condition resources generated")
                    fhir = data["fhir_conditions"][0]
                    assert fhir["resourceType"] == "Condition", "Must be Condition resource"
                    assert "subject" in fhir, "Must have subject"
                    print("   ✓ FHIR Condition structure valid")
                
                # Critical check: Penicillin allergy should affect the recommendation
                all_evidence = str(data)
                if "penicillin" in all_evidence.lower() or "amoxicillin" in all_evidence.lower():
                    print("   ✓ Penicillin allergy recognized in output")
                
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   {response.text}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # ── Test 3: Validate ICD-10 code in response ──────────────
        print("\n3. ICD-10 validation...")
        try:
            # For pneumonia patient, J18 or J22 should be top diagnosis
            if "top_icd10_code" in locals() or "data" in dir():
                top_code = data.get("top_icd10_code", "")
                if top_code.startswith("J1") or top_code.startswith("J2"):
                    print(f"   ✓ Top diagnosis code {top_code} is a respiratory condition (expected)")
                else:
                    print(f"   ⚠️  Top diagnosis code {top_code} — verify this is correct for the test patient")
        except Exception as e:
            print(f"   ⚠️  Could not validate: {e}")
        
        print("\n" + "=" * 60)
        print("Diagnosis Agent test complete!")
        print("=" * 60)
        print("\nNext step: Build Lab Analysis Agent (Agent 3)")


if __name__ == "__main__":
    asyncio.run(test_diagnosis_agent())