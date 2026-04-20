"""
MediTwin Tool Agent — LangGraph ReAct Agent
Port: 8010

This is a CONSERVATIVE tool-calling agent. It uses tools only when needed:

  TRIAGE LOGIC (decided by the LLM via system prompt):
  ┌────────────────────────────────────────────────────────────┐
  │  Query contains a patient ID?                              │
  │                                                            │
  │  YES → fetch_patient_context first, then call ONLY the     │
  │         tools relevant to what the user is asking about    │
  │                                                            │
  │  NO  → answer from general medical knowledge, no tools     │
  └────────────────────────────────────────────────────────────┘

This is intentionally different from the Orchestrator (port 8000):
  - Orchestrator: deterministic, always runs all 8 agents in fixed order
  - Tool Agent:   LLM decides scope — minimal tools for focused queries,
                  zero tools for general medical questions
"""
import os
import sys
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools import MEDITWIN_TOOLS

logger = logging.getLogger("meditwin.tool_agent.agent")


# ── System prompt — triage logic lives here ───────────────────────────────────
SYSTEM_PROMPT = """You are MediTwin AI, a clinical decision support assistant backed by 8 specialist AI agents.

═══════════════════════════════════════════════════════════
CORE TRIAGE RULE — read this carefully before every response
═══════════════════════════════════════════════════════════

CASE A — No patient ID in the query:
  → Answer directly from your clinical knowledge. Do NOT call any tool.
  → Treat this as a general medical question (pharmacology, pathophysiology,
    guidelines, definitions, drug mechanisms, etc.).
  → Be thorough and accurate. You are a knowledgeable clinical assistant.

CASE B — A patient ID is present in the query:
  → ALWAYS start with fetch_patient_context to retrieve the patient's data.
  → After getting the patient data, read the user's query carefully and call
    ONLY the tools that directly answer what is being asked.
  → Do NOT run the full pipeline for every query. Be selective.

═══════════════════════════════════════════════════════════
SELECTIVE TOOL USAGE GUIDE (Case B only)
═══════════════════════════════════════════════════════════

Query asks about...                       → Tools to call
─────────────────────────────────────────────────────────
Diagnosis / what's wrong / ICD-10         → run_diagnosis
Lab results / blood work / abnormals      → analyze_labs
Medications / drug safety / interactions  → check_drug_safety (with proposed meds extracted from query)
X-ray / imaging (+ image provided)        → analyze_chest_xray
Prognosis / treatment comparison          → run_diagnosis + simulate_treatment_outcomes
Full clinical assessment / SOAP note      → run_diagnosis + analyze_labs + run_consensus + generate_clinical_report
Complete workup (explicit request)        → all relevant tools + run_consensus + generate_clinical_report

IMPORTANT — Do NOT call:
  - analyze_chest_xray unless an actual image is in the conversation
  - simulate_treatment_outcomes unless asked about outcomes/prognosis
  - run_consensus unless preparing a final validated report
  - generate_clinical_report unless the user asks for a written report/note

═══════════════════════════════════════════════════════════
TOOL CHAINING RULES
═══════════════════════════════════════════════════════════

1. fetch_patient_context is ALWAYS the first tool call (Case B only)
2. Pass the raw JSON string output of each tool directly into the next tool
3. Never parse or summarize tool output before passing it to another tool
4. proposed_medications for check_drug_safety must be extracted from the user's query
   (e.g. if user asks "is amoxicillin safe?" → proposed_medications=["Amoxicillin"])

═══════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════

After tool calls are complete:
  - Lead with a clear clinical answer to what was asked
  - Highlight critical findings (sepsis risk, UNSAFE drugs, critical labs)
  - If human_review_required is true, always call this out explicitly
  - For general questions (Case A), respond in clear clinical prose

Always be concise. Do not repeat raw JSON in your response.
"""


async def build_tool_agent(checkpointer):
    """
    Build and return the MediTwin Tool Agent with PostgreSQL checkpointer.
    Uses create_react_agent from langgraph.prebuilt (recommended 2025 pattern).
    """
    db_uri = os.getenv(
        "POSTGRES_CHECKPOINT_URI",
        "postgresql://postgres:postgres@postgres-checkpoint:5432/meditwin_checkpoints"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        convert_system_message_to_human=True,
    )

    try:
        await checkpointer.setup()
        logger.info("✓ PostgreSQL checkpointer initialized")
    except Exception as e:
        logger.warning(f"Checkpointer setup: {e} (tables may already exist)")

    agent = create_agent(
        model=llm,
        tools=MEDITWIN_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    logger.info(f"✓ MediTwin Tool Agent built — {len(MEDITWIN_TOOLS)} tools available")
    logger.info("  Conservative mode: tools called only when patient ID present and relevant")
    for t in MEDITWIN_TOOLS:
        logger.info(f"   • {t.name}")

    return agent