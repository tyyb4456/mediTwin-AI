# agents/diagnosis/rag_chain.py
import os
from typing import Optional

import chromadb
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from shared.models import DiagnosisOutput
from dotenv import load_dotenv
load_dotenv()


COLLECTION_NAME = "medical_knowledge"

SYSTEM_PROMPT = """You are a clinical decision support system performing differential diagnosis.
You receive retrieved clinical guidelines and patient data.

Rules:
- Every diagnosis must have a valid ICD-10 code
- Confidence scores must be between 0.0 and 1.0  
- Reasoning must cite specific patient findings from the data provided
- Do not hallucinate drug names or lab values — only use what is in the patient data
- Return 2-4 diagnoses in the differential, most clinically relevant first
"""

HUMAN_PROMPT = """RETRIEVED CLINICAL GUIDELINES:
{context}

PATIENT DATA:
{patient_data}

Return a ranked differential diagnosis based on the above.
"""


class DiagnosisRAG:

    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._structured_llm = None
        self._retrieve_tool = None
        self._initialized = False

    def initialize(self):
        chromadb_host = os.getenv("CHROMADB_HOST", "localhost")
        chromadb_port = int(os.getenv("CHROMADB_PORT", "8008"))

        client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        self._vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        count = self._vectorstore._collection.count()
        if count == 0:
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty. Run ingest.py first.")

        print(f"✓ ChromaDB connected — {count} chunks in '{COLLECTION_NAME}'")

        # ── key change: use with_structured_output instead of JsonOutputParser ──
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
        self._structured_llm = llm.with_structured_output(DiagnosisOutput)

        # ── new docs pattern: @tool with response_format="content_and_artifact" ──
        vectorstore = self._vectorstore  # capture for closure

        @tool(response_format="content_and_artifact")
        def retrieve_clinical_context(query: str):
            """Retrieve clinical guidelines relevant to a patient presentation."""
            docs = vectorstore.similarity_search(query, k=6)
            serialized = "\n\n---\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in docs
            )
            return serialized, docs  # (content for LLM, raw docs as artifact)

        self._retrieve_tool = retrieve_clinical_context
        self._initialized = True
        print("✓ Diagnosis RAG chain initialized")

    def _build_patient_query(self, patient_state: dict, chief_complaint: str) -> str:
        """Same logic as before — unchanged."""
        demographics = patient_state.get("demographics", {})
        age = demographics.get("age", "unknown")
        gender = demographics.get("gender", "unknown")

        conditions = [c.get("display", "") for c in patient_state.get("active_conditions", [])[:5]] or ["None"]
        medications = [m.get("drug", "") for m in patient_state.get("medications", [])[:5]] or ["None"]
        allergies = [
            f"{a.get('substance', '')} ({a.get('severity', '')})"
            for a in patient_state.get("allergies", [])[:3]
        ] or ["None"]
        abnormal_labs = [
            f"{l.get('display', '')}: {l.get('value', '')} {l.get('unit', '')} [{l.get('flag', '')}]"
            for l in patient_state.get("lab_results", [])
            if l.get("flag") in ["HIGH", "LOW", "CRITICAL"]
        ][:8]

        return (
            f"Patient: {age}-year-old {gender}\n"
            f"Chief Complaint: {chief_complaint}\n"
            f"Active Conditions: {', '.join(conditions)}\n"
            f"Current Medications: {', '.join(medications)}\n"
            f"Allergies: {', '.join(allergies)}\n"
            f"Abnormal Labs: {', '.join(abnormal_labs) if abnormal_labs else 'None'}"
        )

    def _apply_rule_adjustments(self, output: DiagnosisOutput, patient_state: dict) -> DiagnosisOutput:
        """Deterministic confidence adjustments on top of LLM output. Same logic, now works on Pydantic model."""
        allergies = {a.get("substance", "").lower() for a in patient_state.get("allergies", [])}
        labs = {l.get("loinc"): l for l in patient_state.get("lab_results", [])}

        wbc_flag = labs.get("26464-8", {}).get("flag", "")
        crp_flag = labs.get("1988-5", {}).get("flag", "")

        for diag in output.differential_diagnosis:
            if diag.icd10_code.startswith("J1"):
                if wbc_flag in ["HIGH", "CRITICAL"] and crp_flag in ["HIGH", "CRITICAL"]:
                    diag.confidence = round(min(diag.confidence + 0.08, 0.97), 2)
                if "penicillin" in allergies or "amoxicillin" in allergies:
                    note = "Penicillin allergy — standard beta-lactam therapy contraindicated"
                    if note not in diag.against_evidence:
                        diag.against_evidence.append(note)

        # Recompute confidence_level from top diagnosis
        top_conf = output.differential_diagnosis[0].confidence if output.differential_diagnosis else 0
        if top_conf >= 0.80:
            output.confidence_level = "HIGH"
        elif top_conf >= 0.60:
            output.confidence_level = "MODERATE"
        else:
            output.confidence_level = "LOW"

        return output

    def run(self, patient_state: dict, chief_complaint: str) -> DiagnosisOutput:
        if not self._initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        patient_query = self._build_patient_query(patient_state, chief_complaint)

        # Retrieve context using the tool directly
        docs = self._vectorstore.similarity_search(patient_query, k=6)
        context = "\n\n---\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        # Chain: prompt | structured LLM → DiagnosisOutput Pydantic object directly
        chain = prompt | self._structured_llm
        output: DiagnosisOutput = chain.invoke({
            "context": context,
            "patient_data": patient_query,
        })

        output = self._apply_rule_adjustments(output, patient_state)

        # Sync top_diagnosis/code from rank 1
        if output.differential_diagnosis:
            top = output.differential_diagnosis[0]
            output.top_diagnosis = top.display
            output.top_icd10_code = top.icd10_code

        return output

    def build_fhir_conditions(self, output: DiagnosisOutput, patient_id: str) -> list[dict]:
        """Now takes DiagnosisOutput directly instead of a raw dict."""
        conditions = []
        for diag in output.differential_diagnosis[:3]:
            conditions.append({
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                                      "code": diag.icd10_code, "display": diag.display}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                                    "code": "provisional"}]},
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                                "code": "active"}]},
                "note": [{"text": (
                    f"AI differential (rank {diag.rank}). "
                    f"Confidence: {diag.confidence:.0%}. "
                    f"{diag.clinical_reasoning}. "
                    "Requires physician verification."
                )}]
            })
        return conditions


diagnosis = DiagnosisRAG()