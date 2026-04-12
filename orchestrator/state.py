"""
MediTwin State — LangGraph StateGraph definition
The canonical shared state object that every node reads from and writes to.

Design rules (from strategy doc):
  - Each node only writes to its OWN designated fields
  - No node writes to another node's output fields
  - All values must be JSON-serializable (no NumPy arrays, no TF tensors)
  - Optional fields use None default — agents that don't run leave None
"""
from typing import Optional, Annotated
from typing import TypedDict
import operator


class MediTwinState(TypedDict):
    # ── Input (set by Orchestrator at graph entry) ─────────────────────────────
    patient_id: str
    chief_complaint: str
    fhir_base_url: str
    sharp_token: Optional[str]                    # SHARP bearer token (may be empty)
    image_data: Optional[str]           # base64 chest X-ray (optional)
    imaging_available: bool             # derived from image_data presence

    # ── Agent outputs (populated as graph executes) ────────────────────────────
    patient_state: Optional[dict]       # Agent 1: Patient Context
    diagnosis_output: Optional[dict]    # Agent 2: Diagnosis
    lab_output: Optional[dict]          # Agent 3: Lab Analysis
    drug_safety_output: Optional[dict]  # Agent 4: Drug Safety
    imaging_output: Optional[dict]      # Agent 5: Imaging Triage
    digital_twin_output: Optional[dict] # Agent 6: Digital Twin
    consensus_output: Optional[dict]    # Agent 7: Consensus
    final_output: Optional[dict]        # Agent 8: Explanation

    # ── Control flags ─────────────────────────────────────────────────────────
    human_review_required: bool
    error_log: Annotated[list, operator.add]   # append-only: each node can add errors