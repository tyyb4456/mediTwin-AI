"""
models.py — Pydantic request/response models for Digital Twin Agent
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class TreatmentOption(BaseModel):
    option_id: str = Field(description="Short identifier: 'A', 'B', 'C'")
    label: str = Field(description="Human-readable label")
    drugs: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    estimated_cost_usd: Optional[float] = None
    requires_hospitalization: bool = False
    expected_duration_days: Optional[int] = None


class DigitalTwinRequest(BaseModel):
    patient_state: dict
    diagnosis: str = Field(default="Unknown diagnosis")
    diagnosis_code: Optional[str] = None  # ICD-10 for guideline lookup
    treatment_options: List[TreatmentOption] = Field(default_factory=list)
    include_sensitivity_analysis: bool = True
    include_cost_analysis: bool = True
    prediction_horizons: List[str] = Field(default=["7d", "30d", "90d"])  # Configurable
    patient_preferences: Optional[Dict] = None  # For shared decision making


class SensitivityAnalysis(BaseModel):
    """Results of one-way sensitivity analysis on key risk factors"""
    feature_name: str
    baseline_value: float
    risk_impact_if_improved_10_percent: dict
    risk_impact_if_worsened_10_percent: dict
    modifiable: bool
    clinical_intervention: Optional[str] = None


class PredictionWithUncertainty(BaseModel):
    """Risk prediction with Bayesian credible intervals"""
    point_estimate: float
    lower_bound_95ci: float
    upper_bound_95ci: float
    confidence_level: str  # HIGH, MODERATE, LOW based on interval width


class ScenarioComparison(BaseModel):
    """Enhanced scenario with multi-horizon predictions and uncertainty"""
    option_id: str
    label: str
    drugs: List[str]
    interventions: List[str]
    predictions_7d: Dict[str, PredictionWithUncertainty]
    predictions_30d: Dict[str, PredictionWithUncertainty]
    predictions_90d: Optional[Dict[str, PredictionWithUncertainty]] = None
    key_risks: List[str]
    guideline_adherence: Optional[Dict] = None
    cost_effectiveness: Optional[Dict] = None
    adherence_adjusted_efficacy: float = 1.0  # 0.0-1.0


class DigitalTwinResponse(BaseModel):
    simulation_summary: dict
    scenarios: List[dict]
    what_if_narrative: str
    fhir_care_plan: Optional[dict]
    feature_attribution: List[dict]
    sensitivity_analysis: Optional[List[SensitivityAnalysis]] = None
    cost_effectiveness_summary: Optional[Dict] = None
    models_loaded: bool
    model_confidence: str  # Overall model confidence
    provenance: dict  # Tracking for reproducibility
    mock: bool = False