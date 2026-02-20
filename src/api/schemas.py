"""
API schemas and validation models
Centralized location for all Pydantic models
"""
from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class PredictionRequest(BaseModel):
    """
    Request model for heart disease prediction with medical reality checks.
    Ranges are widened to accept medically possible extreme values while rejecting physical impossibilities.
    """
    age: float = Field(..., ge=1, le=120, description="Age in years (1-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (50-300 mm Hg)")
    chol: float = Field(..., ge=50, le=800, description="Serum cholesterol (50-800 mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0: false, 1: true)")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Maximum heart rate achieved (30-250)")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0: no, 1: yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression (0-10)")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    @model_validator(mode="after")
    def validate_hemodynamic_consistency(self) -> "PredictionRequest":
        """Cross-field plausibility checks for medical safety."""
        pulse_pressure = self.trestbps - self.oldpeak * 10
        if pulse_pressure < 30:
            raise ValueError("Hemodynamic pattern is inconsistent: very low inferred pulse pressure")

        if self.age < 18 and self.ca > 2:
            raise ValueError("Major vessel count is implausible for pediatric age")

        return self
    
    @field_validator('thalach')
    @classmethod
    def validate_heart_rate(cls, v: float, info: ValidationInfo) -> float:
        """Validate heart rate is reasonable for age"""
        if not info.data:
            return v
            
        age = info.data.get('age')
        if age is not None:
            # Theoretical max heart rate is approx 220 - age.
            # We allow significant margin (10%) for outliers.
            max_hr = 220 - age
            limit = max_hr * 1.1
            if v > limit:
                raise ValueError(f'Heart rate {v} seems too high for age {age} (Max expected ~{int(limit)})')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
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
        description="Top features influencing the prediction (proxy explanation)."
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[PredictionRequest] = Field(
        ..., 
        min_items=1,
        max_items=100, 
        description="List of prediction requests (max 100)"
    )

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total: int
    timestamp: str

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
