"""
Test script for Patient Context Agent
Tests against HAPI FHIR public sandbox
"""
import asyncio
import httpx


async def test_patient_context_agent():
    """Test the patient context agent"""
    
    # Test patient ID from HAPI FHIR sandbox
    # This is a well-known test patient
    patient_id = "example"  # Standard test patient in HAPI FHIR
    
    agent_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Testing Patient Context Agent...")
        print(f"Patient ID: {patient_id}")
        print("-" * 50)
        
        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        try:
            response = await client.get(f"{agent_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   ERROR: {e}")
            return
        
        # Test fetch endpoint
        print("\n2. Testing /fetch endpoint...")
        try:
            response = await client.post(
                f"{agent_url}/fetch",
                json={
                    "patient_id": patient_id,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                patient_state = data["patient_state"]
                
                print(f"\n   ✓ Patient fetched successfully!")
                print(f"   Cache hit: {data['cache_hit']}")
                print(f"   Fetch time: {data['fetch_time_ms']}ms")
                print(f"\n   Demographics:")
                print(f"      Name: {patient_state['demographics']['name']}")
                print(f"      Age: {patient_state['demographics']['age']}")
                print(f"      Gender: {patient_state['demographics']['gender']}")
                print(f"\n   Active Conditions: {len(patient_state['active_conditions'])}")
                for cond in patient_state['active_conditions'][:3]:
                    print(f"      - {cond['display']} ({cond['code']})")
                
                print(f"\n   Medications: {len(patient_state['medications'])}")
                for med in patient_state['medications'][:3]:
                    print(f"      - {med['drug']}")
                
                print(f"\n   Allergies: {len(patient_state['allergies'])}")
                for allergy in patient_state['allergies'][:3]:
                    print(f"      - {allergy['substance']}")
                
                print(f"\n   Lab Results: {len(patient_state['lab_results'])}")
                for lab in patient_state['lab_results'][:5]:
                    print(f"      - {lab['display']}: {lab['value']} {lab['unit']} [{lab['flag']}]")
            else:
                print(f"   ERROR: {response.text}")
        
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Test cache hit
        print("\n3. Testing cache (should be faster)...")
        try:
            response = await client.post(
                f"{agent_url}/fetch",
                json={
                    "patient_id": patient_id,
                    "fhir_base_url": "https://hapi.fhir.org/baseR4"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ Cache hit: {data['cache_hit']}")
                print(f"   Fetch time: {data['fetch_time_ms']}ms (should be < 10ms)")
            
        except Exception as e:
            print(f"   ERROR: {e}")
        
        print("\n" + "=" * 50)
        print("Testing complete!")


if __name__ == "__main__":
    asyncio.run(test_patient_context_agent())