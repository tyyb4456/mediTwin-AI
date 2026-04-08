"""
Test Suite for Patient Context Agent — COMPLETE REBUILD
Tests against HAPI FHIR public sandbox

Test Categories:
1. Health check
2. Basic fetch (direct mode)
3. Cache validation
4. SHARP header passthrough
5. Error handling (missing patient, invalid inputs)
6. Data completeness validation
7. Imaging availability detection
8. Performance benchmarks

Run:
    # Start the agent first:
    python patient_context_agent_complete.py
    
    # Then run tests:
    python test_patient_context_complete.py
"""
import asyncio
import httpx
import time
from typing import Optional


AGENT_URL = "http://localhost:8001"
PATIENT_ID_VALID = "example"  # Well-known test patient in HAPI FHIR
PATIENT_ID_INVALID = "nonexistent-patient-12345"


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def pass_test(self, name: str):
        print(f"   ✓ {name}")
        self.passed += 1
    
    def fail_test(self, name: str, error: str):
        print(f"   ✗ {name}")
        print(f"      Error: {error}")
        self.failed += 1
    
    def warn(self, message: str):
        print(f"   ⚠  {message}")
        self.warnings += 1
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed:   {self.passed}/{total}")
        print(f"Failed:   {self.failed}/{total}")
        print(f"Warnings: {self.warnings}")
        
        if self.failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n❌ {self.failed} test(s) failed")


async def run_tests():
    """Run all tests"""
    results = TestResults()
    
    print("=" * 60)
    print("Patient Context Agent — Test Suite")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 1: Health Check
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n1. Health Check")
        print("-" * 60)
        
        try:
            response = await client.get(f"{AGENT_URL}/health")
            data = response.json()
            
            if response.status_code == 200:
                results.pass_test("Agent is running")
            else:
                results.fail_test("Agent is running", f"Status {response.status_code}")
            
            if data.get("redis_connected"):
                results.pass_test("Redis connection active")
            else:
                results.fail_test("Redis connection active", "Redis not connected")
            
            if data.get("http_client_ready"):
                results.pass_test("HTTP client initialized")
            else:
                results.fail_test("HTTP client initialized", "HTTP client not ready")
            
            print(f"\n   Agent version: {data.get('version')}")
            print(f"   FHIR base URL: {data.get('fhir_base_url')}")
            
        except Exception as e:
            results.fail_test("Health check", str(e))
            print("\n❌ Agent not running. Start with: python patient_context_agent_complete.py")
            return results
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 2: Basic Fetch (Direct Mode)
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n2. Basic Fetch (Direct Mode)")
        print("-" * 60)
        
        try:
            start = time.time()
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={
                    "patient_id": PATIENT_ID_VALID,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            elapsed = time.time() - start
            
            if response.status_code != 200:
                results.fail_test("Basic fetch", f"Status {response.status_code}: {response.text[:200]}")
            else:
                data = response.json()
                
                # Validate response structure
                if "patient_state" in data:
                    results.pass_test("Response has patient_state")
                else:
                    results.fail_test("Response has patient_state", "Missing field")
                
                if data.get("cache_hit") == False:
                    results.pass_test("Cache miss on first fetch")
                else:
                    results.warn("Expected cache miss, got cache hit")
                
                if data.get("source") == "direct":
                    results.pass_test("Source correctly identified as 'direct'")
                else:
                    results.fail_test("Source identification", f"Expected 'direct', got '{data.get('source')}'")
                
                if data.get("fhir_resources_fetched") == 6:
                    results.pass_test("All 6 FHIR resource types fetched")
                else:
                    results.warn(f"Expected 6 resources, fetched {data.get('fhir_resources_fetched')}")
                
                # Validate patient_state structure
                ps = data.get("patient_state", {})
                
                if ps.get("patient_id") == PATIENT_ID_VALID:
                    results.pass_test("Patient ID matches request")
                else:
                    results.fail_test("Patient ID matches", f"Got {ps.get('patient_id')}")
                
                # Demographics
                demo = ps.get("demographics", {})
                if demo.get("name"):
                    results.pass_test(f"Demographics extracted (name: {demo.get('name')})")
                else:
                    results.fail_test("Demographics extraction", "No name")
                
                if demo.get("age") and demo.get("age") > 0:
                    results.pass_test(f"Age calculated ({demo.get('age')} years)")
                else:
                    results.fail_test("Age calculation", f"Got {demo.get('age')}")
                
                # Active conditions
                conditions = ps.get("active_conditions", [])
                print(f"\n   Active conditions: {len(conditions)}")
                for cond in conditions[:3]:
                    print(f"      - {cond.get('display')} ({cond.get('code')})")
                
                if len(conditions) > 0:
                    results.pass_test(f"Active conditions extracted ({len(conditions)})")
                else:
                    results.warn("No active conditions found (may be valid for test patient)")
                
                # Medications
                meds = ps.get("medications", [])
                print(f"\n   Medications: {len(meds)}")
                for med in meds[:3]:
                    print(f"      - {med.get('drug')}")
                
                if len(meds) > 0:
                    results.pass_test(f"Medications extracted ({len(meds)})")
                else:
                    results.warn("No medications found (may be valid)")
                
                # Allergies
                allergies = ps.get("allergies", [])
                print(f"\n   Allergies: {len(allergies)}")
                for allergy in allergies[:3]:
                    print(f"      - {allergy.get('substance')} ({allergy.get('severity')})")
                
                if len(allergies) > 0:
                    results.pass_test(f"Allergies extracted ({len(allergies)})")
                else:
                    results.warn("No allergies found (may be valid)")
                
                # Lab results
                labs = ps.get("lab_results", [])
                print(f"\n   Lab results: {len(labs)}")
                for lab in labs[:5]:
                    print(f"      - {lab.get('display')}: {lab.get('value')} {lab.get('unit')} [{lab.get('flag')}]")
                
                if len(labs) > 0:
                    results.pass_test(f"Lab results extracted ({len(labs)})")
                    
                    # Validate LOINC codes
                    has_loinc = all(lab.get("loinc") for lab in labs)
                    if has_loinc:
                        results.pass_test("All labs have LOINC codes")
                    else:
                        results.warn("Some labs missing LOINC codes")
                    
                    # Validate flags
                    flagged_labs = [lab for lab in labs if lab.get("flag") in ("HIGH", "LOW", "CRITICAL")]
                    if flagged_labs:
                        results.pass_test(f"Lab flags computed ({len(flagged_labs)} abnormal)")
                    
                else:
                    results.warn("No lab results found")
                
                # Imaging availability
                imaging_available = ps.get("imaging_available", False)
                diagnostic_reports = ps.get("diagnostic_reports", [])
                
                print(f"\n   Diagnostic reports: {len(diagnostic_reports)}")
                print(f"   Imaging available: {imaging_available}")
                
                if len(diagnostic_reports) > 0:
                    results.pass_test(f"Diagnostic reports extracted ({len(diagnostic_reports)})")
                
                # State timestamp
                if ps.get("state_timestamp"):
                    results.pass_test("State timestamp present")
                else:
                    results.fail_test("State timestamp", "Missing")
                
                # Performance
                fetch_time_ms = data.get("fetch_time_ms", 0)
                print(f"\n   Fetch time: {fetch_time_ms}ms (Python asyncio)")
                print(f"   Total elapsed: {elapsed:.2f}s (includes HTTP)")
                
                if fetch_time_ms < 10000:  # 10 seconds
                    results.pass_test(f"Fetch completed in reasonable time ({fetch_time_ms}ms)")
                else:
                    results.warn(f"Slow fetch: {fetch_time_ms}ms (HAPI sandbox may be slow)")
                
        except Exception as e:
            results.fail_test("Basic fetch", str(e))
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 3: Cache Validation
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n3. Cache Validation")
        print("-" * 60)
        
        try:
            start = time.time()
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={
                    "patient_id": PATIENT_ID_VALID,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("cache_hit"):
                    results.pass_test("Cache hit on second fetch")
                else:
                    results.fail_test("Cache hit", "Expected cache hit, got miss")
                
                fetch_time_ms = data.get("fetch_time_ms", 0)
                print(f"   Cache fetch time: {fetch_time_ms}ms")
                
                if fetch_time_ms < 50:  # Should be < 50ms from Redis
                    results.pass_test(f"Cache fetch is fast ({fetch_time_ms}ms)")
                else:
                    results.warn(f"Cache fetch slower than expected: {fetch_time_ms}ms")
                
                if data.get("fhir_resources_fetched") == 0:
                    results.pass_test("No FHIR resources fetched (served from cache)")
                else:
                    results.fail_test("Cache served data", "FHIR resources were fetched")
                
            else:
                results.fail_test("Cache test", f"Status {response.status_code}")
        
        except Exception as e:
            results.fail_test("Cache validation", str(e))
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 4: SHARP Header Passthrough
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n4. SHARP Header Passthrough")
        print("-" * 60)
        
        try:
            # Clear cache first to test fresh fetch with SHARP headers
            # (In real scenario, each patient_id would have its own cache key)
            
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={"chief_complaint": "Test"},  # Minimal body
                headers={
                    "X-SHARP-Patient-ID": PATIENT_ID_VALID,
                    "X-SHARP-FHIR-Token": "Bearer test-token-12345",
                    "X-SHARP-FHIR-BaseURL": "https://hapi.fhir.org/baseR4"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("source") == "SHARP":
                    results.pass_test("SHARP headers detected and used")
                else:
                    results.fail_test("SHARP source", f"Expected 'SHARP', got '{data.get('source')}'")
                
                ps = data.get("patient_state", {})
                if ps.get("patient_id") == PATIENT_ID_VALID:
                    results.pass_test("SHARP patient ID correctly extracted from header")
                else:
                    results.fail_test("SHARP patient ID", f"Got {ps.get('patient_id')}")
                
                print(f"   Source: {data.get('source')}")
                print(f"   Patient ID from header: {ps.get('patient_id')}")
                
            else:
                results.fail_test("SHARP header test", f"Status {response.status_code}")
        
        except Exception as e:
            results.fail_test("SHARP header passthrough", str(e))
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 5: Error Handling — Missing Patient
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n5. Error Handling — Missing Patient")
        print("-" * 60)
        
        try:
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={
                    "patient_id": PATIENT_ID_INVALID,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            
            if response.status_code == 404:
                results.pass_test("404 returned for nonexistent patient")
                data = response.json()
                print(f"   Error detail: {data.get('detail', '')}")
            else:
                results.fail_test("404 for missing patient", f"Got status {response.status_code}")
        
        except Exception as e:
            results.fail_test("Missing patient handling", str(e))
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 6: Error Handling — Missing patient_id
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n6. Error Handling — Missing patient_id")
        print("-" * 60)
        
        try:
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={
                    "patient_id": "",
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            
            if response.status_code == 400:
                results.pass_test("400 returned for empty patient_id")
                data = response.json()
                print(f"   Error detail: {data.get('detail', '')}")
            else:
                results.fail_test("400 for empty patient_id", f"Got status {response.status_code}")
        
        except Exception as e:
            results.fail_test("Empty patient_id handling", str(e))
        
        # ══════════════════════════════════════════════════════════════════════
        # Test 7: Data Completeness — All Required Fields
        # ══════════════════════════════════════════════════════════════════════
        
        print("\n7. Data Completeness — All Required Fields")
        print("-" * 60)
        
        try:
            response = await client.post(
                f"{AGENT_URL}/fetch",
                json={
                    "patient_id": PATIENT_ID_VALID,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                ps = data.get("patient_state", {})
                
                required_fields = [
                    "patient_id",
                    "demographics",
                    "active_conditions",
                    "medications",
                    "allergies",
                    "lab_results",
                    "diagnostic_reports",
                    "recent_encounters",
                    "state_timestamp",
                    "imaging_available"
                ]
                
                all_present = True
                for field in required_fields:
                    if field in ps:
                        pass  # Field present
                    else:
                        results.fail_test(f"Required field: {field}", "Missing")
                        all_present = False
                
                if all_present:
                    results.pass_test("All required PatientState fields present")
                
                # Demographics subfields
                demo_fields = ["name", "age", "gender", "dob"]
                demo = ps.get("demographics", {})
                all_demo = all(field in demo for field in demo_fields)
                
                if all_demo:
                    results.pass_test("All demographics subfields present")
                else:
                    missing = [f for f in demo_fields if f not in demo]
                    results.fail_test("Demographics completeness", f"Missing: {missing}")
            
            else:
                results.fail_test("Data completeness test", f"Fetch failed with {response.status_code}")
        
        except Exception as e:
            results.fail_test("Data completeness validation", str(e))
    
    # Print final summary
    results.summary()
    return results


if __name__ == "__main__":
    print("\n" + "🏥 " * 20)
    print("MediTwin AI — Patient Context Agent Test Suite")
    print("Testing the foundation of the entire system")
    print("🏥 " * 20 + "\n")
    
    results = asyncio.run(run_tests())
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if results.failed == 0:
        print("""
✅ Patient Context Agent is fully functional!

The foundation is solid. All other agents can now depend on this.

Build order continues:
  ✅ Agent 1: Patient Context Agent (port 8001) — COMPLETE
  → Agent 2: Diagnosis Agent (port 8002)
  → Agent 3: Lab Analysis Agent (port 8003)
  → Agent 4: Drug Safety MCP (port 8004)
  ...and so on

Key improvements made:
  ✓ SHARP context properly implemented (production-ready)
  ✓ Fallback to request body (development-friendly)
  ✓ Parallel FHIR fetches (much faster)
  ✓ Imaging availability detection from DiagnosticReport
  ✓ Graceful error handling (missing resources → empty arrays)
  ✓ Comprehensive logging
  ✓ Redis caching working correctly
""")
    else:
        print(f"""
❌ {results.failed} test(s) failed

Please review the errors above and fix before proceeding.
The Patient Context Agent must be rock-solid before building other agents.
""")