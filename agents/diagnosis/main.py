"""
Agent 2: Diagnosis Agent
RAG-based differential diagnosis over medical knowledge base.
Port: 8002
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directories to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.models import PatientState

# Import the RAG chain (singleton)
from agents.rag import diagnosis


# ── Request / Response Models ──────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    patient_state: dict  # Full PatientState dict from Patient Context Agent
    chief_complaint: str
    include_fhir_resources: bool = True


class DiagnoseResponse(BaseModel):
    differential_diagnosis: list
    top_diagnosis: str
    top_icd10_code: str
    confidence_level: str
    reasoning_summary: str
    recommended_next_steps: list
    fhir_conditions: Optional[list] = None  # FHIR Condition resources
    rag_available: bool  # Whether RAG was used or fallback


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain at startup"""
    global rag_ready
    
    try:
        diagnosis.initialize()
        rag_ready = True
        print("✓ Diagnosis Agent started — RAG chain ready")
    except Exception as e:
        rag_ready = False
        print(f"⚠️  Diagnosis Agent started — RAG unavailable: {e}")
        print("   Agent will use fallback mode (LLM only, no RAG context)")
    
    yield
    
    print("✓ Diagnosis Agent shutdown")


rag_ready = False

app = FastAPI(
    title="MediTwin Diagnosis Agent",
    description="RAG-based differential diagnosis from patient FHIR data",
    version="1.0.0",
    lifespan=lifespan
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    if not request.patient_state:
        raise HTTPException(status_code=400, detail="patient_state is required")
    if not request.chief_complaint:
        raise HTTPException(status_code=400, detail="chief_complaint is required")

    patient_id = request.patient_state.get("patient_id", "unknown")

    try:
        if rag_ready:
            result = diagnosis.run(
                patient_state=request.patient_state,
                chief_complaint=request.chief_complaint
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Diagnosis Agent not ready. ChromaDB may be unavailable or not seeded."
            )

        fhir_conditions = None
        if request.include_fhir_resources:
            fhir_conditions = diagnosis.build_fhir_conditions(result, patient_id)

        return DiagnoseResponse(
            differential_diagnosis=[diag.model_dump() for diag in result.differential_diagnosis],
            top_diagnosis=result.top_diagnosis,
            top_icd10_code=result.top_icd10_code,
            confidence_level=result.confidence_level,
            reasoning_summary=result.reasoning_summary,
            recommended_next_steps=result.recommended_next_steps,
            fhir_conditions=fhir_conditions,
            rag_available=rag_ready
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check — also reports RAG chain status"""
    return {
        "status": "healthy",
        "agent": "diagnosis",
        "version": "1.0.0",
        "rag_ready": rag_ready,
        "chromadb_collection": "medical_knowledge" if rag_ready else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)