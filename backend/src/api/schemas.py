"""
API schemas and validation models
Centralized location for all Pydantic models
"""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import yaml

def _load_validation_ranges() -> Dict[str, Dict[str, float]]:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("validation", {}).get("ranges", {})
    except Exception:
        return {}

_RANGES = _load_validation_ranges()

def _rng(name: str, default_min: float, default_max: float) -> Dict[str, float]:
    values = _RANGES.get(name, {})
    return {"min": values.get("min", default_min), "max": values.get("max", default_max)}

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class PredictionRequest(BaseModel):
    """
    Request model for cardiovascular risk prediction, based on the Kaggle
    cardiovascular lifestyle dataset's 11 features (age, sex, BMI inputs,
    blood pressure, cholesterol/glucose category, and lifestyle factors).
    Ranges are widened to accept medically possible extreme values while
    rejecting physical impossibilities. BMI is derived server-side from
    height/weight rather than accepted directly, to avoid client/server
    disagreement about the derived value.
    """
    age: float = Field(..., ge=_rng("age", 18, 100)["min"], le=_rng("age", 18, 100)["max"], description="Age in years")
    sex: int = Field(..., ge=_rng("sex", 0, 1)["min"], le=_rng("sex", 0, 1)["max"], description="Sex (0: female, 1: male)")
    height: float = Field(..., ge=_rng("height", 120, 220)["min"], le=_rng("height", 120, 220)["max"], description="Height (cm)")
    weight: float = Field(..., ge=_rng("weight", 30, 250)["min"], le=_rng("weight", 30, 250)["max"], description="Weight (kg)")
    ap_hi: float = Field(..., ge=_rng("ap_hi", 70, 240)["min"], le=_rng("ap_hi", 70, 240)["max"], description="Systolic blood pressure (mm Hg)")
    ap_lo: float = Field(..., ge=_rng("ap_lo", 40, 160)["min"], le=_rng("ap_lo", 40, 160)["max"], description="Diastolic blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=_rng("cholesterol", 1, 3)["min"], le=_rng("cholesterol", 1, 3)["max"], description="Cholesterol level (1: normal, 2: above normal, 3: well above normal)")
    gluc: int = Field(..., ge=_rng("gluc", 1, 3)["min"], le=_rng("gluc", 1, 3)["max"], description="Glucose level (1: normal, 2: above normal, 3: well above normal)")
    smoke: int = Field(..., ge=_rng("smoke", 0, 1)["min"], le=_rng("smoke", 0, 1)["max"], description="Current smoker (0: no, 1: yes)")
    alco: int = Field(..., ge=_rng("alco", 0, 1)["min"], le=_rng("alco", 0, 1)["max"], description="Alcohol intake (0: no, 1: yes)")
    active: int = Field(..., ge=_rng("active", 0, 1)["min"], le=_rng("active", 0, 1)["max"], description="Physically active (0: no, 1: yes)")

    @model_validator(mode="after")
    def validate_hemodynamic_consistency(self) -> "PredictionRequest":
        """Cross-field plausibility check for medical safety: systolic must
        meaningfully exceed diastolic blood pressure."""
        if self.ap_hi - self.ap_lo < 10:
            raise ValueError(
                "Hemodynamic pattern is inconsistent: systolic blood pressure "
                "must be at least 10 mmHg above diastolic"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58,
                "sex": 1,
                "height": 175,
                "weight": 85,
                "ap_hi": 145,
                "ap_lo": 90,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Binary prediction (0: no disease, 1: disease)")
    probability: float = Field(..., ge=0, le=1, description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    confidence: float = Field(..., description="Model confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")
    request_id: Optional[str] = Field(None, description="Unique trace ID for the request")
    top_contributors: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Top features driving this prediction, as signed SHAP values "
        "(positive increases predicted risk, negative is protective). None if "
        "no explainer is available for the loaded model or explanation failed."
    )
    baseline_probability: Optional[float] = Field(
        default=None,
        description="SHAP expected value: the model's average predicted probability "
        "over the background dataset, i.e. the starting point before this patient's "
        "specific feature values are applied."
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of prediction requests (max 100)"
    )

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    errors: Optional[List[Dict[str, Any]]] = None
    total: int
    timestamp: str
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_path: Optional[str]
    loaded: bool
    loaded_at: Optional[str]
    version: Optional[str]
    features: List[str]
    training_metadata: Optional[Dict[str, Any]] = None
