"""
Tiebreaker RAG — Consensus Agent
Queries the same ChromaDB medical knowledge collection as the Diagnosis Agent
to resolve MODERATE conflicts that don't require human escalation.

Only invoked when route_consensus() returns "resolve".
Never invoked for HIGH severity conflicts — those go straight to escalation.
"""
import os
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from conflict_detector import Conflict

COLLECTION_NAME = "medical_knowledge"

TIEBREAKER_SYSTEM = """You are a senior clinical decision support specialist resolving a diagnostic conflict.
You have retrieved relevant clinical guidelines to help arbitrate between two conflicting agent outputs.
Return ONLY valid JSON. No preamble, no markdown."""

TIEBREAKER_USER = """RETRIEVED CLINICAL GUIDELINES:
{context}

CONFLICT DETAILS:
Type: {conflict_type}
Description: {conflict_description}

Agent A ({agent_a}) says: {output_a} (confidence: {confidence_a:.0%})
Agent B ({agent_b}) says: {output_b} (confidence: {confidence_b:.0%})

Patient summary: {patient_summary}

Based strictly on the clinical guidelines provided, resolve this conflict.
Return JSON:
{{
  "resolved_diagnosis": "ICD-10 code of the correct diagnosis",
  "resolved_diagnosis_display": "Display name",
  "resolution_method": "TIEBREAKER_RAG",
  "reasoning": "One sentence explaining which agent is correct and why, citing specific clinical evidence",
  "confidence_after_resolution": 0.60 to 0.85,
  "supporting_guideline": "One phrase from the guidelines that supports this resolution"
}}
Return ONLY JSON."""


class TiebreakerRAG:
    """RAG-based conflict resolution. Initialized lazily to avoid startup failures."""

    def __init__(self):
        self._ready = False
        self._vectorstore: Optional[Chroma] = None
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def initialize(self) -> bool:
        """
        Connect to ChromaDB and init LLM.
        Returns True on success.
        Called lazily on first use.
        """
        chromadb_host = os.getenv("CHROMADB_HOST", "localhost")
        chromadb_port = int(os.getenv("CHROMADB_PORT", "8008"))


        try:
            client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
            client.heartbeat()

            embedding_model =GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

            self._vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model,
            )

            count = self._vectorstore._collection.count()
            if count == 0:
                logging.warning(f"  ⚠ Tiebreaker: ChromaDB collection empty — run knowledge_base/ingest.py")
                return False

            self._llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

            self._ready = True
            logging.info(f"  ✔  Tiebreaker RAG ready ({count} chunks in collection)")
            return True

        except Exception as e:
            logging.error(f"  ✘  Tiebreaker RAG init failed: {e}")
            return False

    def _build_patient_summary(self, patient_state: dict) -> str:
        """Extract compact patient summary for tiebreaker context."""
        demo = patient_state.get("demographics", {})
        conditions = patient_state.get("active_conditions", [])
        labs = patient_state.get("lab_results", [])

        abnormal_labs = [
            f"{l['display']}: {l['value']} {l['unit']} [{l['flag']}]"
            for l in labs if l.get("flag") in ("HIGH", "LOW", "CRITICAL")
        ][:5]

        return (
            f"{demo.get('age', '?')}y {demo.get('gender', '?')}. "
            f"Conditions: {', '.join(c['display'] for c in conditions[:3]) or 'None'}. "
            f"Abnormal labs: {', '.join(abnormal_labs) or 'None'}."
        )

    def resolve(self, conflict: Conflict, patient_state: dict) -> Optional[dict]:
        """
        Run RAG tiebreaker for a single conflict.
        Returns resolution dict or None if unavailable.
        """
        if not self._ready:
            if not self.initialize():
                return None

        try:
            # Build query from conflict context
            query = (
                f"{conflict.description} "
                f"Patient has {conflict.output_a} vs {conflict.output_b}. "
                "Which diagnosis is supported by clinical evidence?"
            )

            retriever = self._vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            patient_summary = self._build_patient_summary(patient_state)

            prompt = ChatPromptTemplate.from_messages([
                ("system", TIEBREAKER_SYSTEM),
                ("human", TIEBREAKER_USER),
            ])

            chain = prompt | self._llm | JsonOutputParser()

            result = chain.invoke({
                "context": context,
                "conflict_type": conflict.type,
                "conflict_description": conflict.description,
                "agent_a": conflict.agent_a,
                "output_a": conflict.output_a,
                "confidence_a": conflict.confidence_a,
                "agent_b": conflict.agent_b,
                "output_b": conflict.output_b,
                "confidence_b": conflict.confidence_b,
                "patient_summary": patient_summary,
            })

            return result

        except Exception as e:
            logging.error(f"  ✘  Tiebreaker RAG failed: {e}")
            return None


# Module-level singleton
tiebreaker = TiebreakerRAG()