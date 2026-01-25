"""
API schemas and validation models
Centralized location for all Pydantic models
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class Sex(int, Enum):
    """Sex enumeration"""
    FEMALE = 0
    MALE = 1

class ChestPainType(int, Enum):
    """Chest pain type enumeration"""
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3

class PredictionRequest(BaseModel):
    """Request model for heart disease prediction with comprehensive validation"""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia")
    
    @validator('thalach')
    def validate_heart_rate(cls, v, values):
        """Validate heart rate is reasonable for age"""
        if 'age' in values:
            max_hr = 220 - values['age']
            if v > max_hr * 1.1:  # Allow 10% margin
                raise ValueError(f'Heart rate {v} seems too high for age {values["age"]}')
        return v
    
    @validator('chol')
    def validate_cholesterol(cls, v):
        """Warn about extreme cholesterol values"""
        if v > 500:
            # In production, you might want to log this
            pass
        return v
    
    class Config:
        schema_extra = {
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
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[PredictionRequest] = Field(
        ..., 
        max_items=100, 
        description="List of prediction requests (max 100)"
    )
    
    @validator('instances')
    def validate_instances(cls, v):
        if len(v) == 0:
            raise ValueError('At least one instance is required')
        return v

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    total_requested: int
    success_rate: float
    timestamp: str
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_metadata: Dict[str, Any]
    timestamp: str
    uptime_seconds: Optional[float] = None

class ModelInfo(BaseModel):
    """Model information response"""
    model_path: Optional[str]
    loaded: bool
    loaded_at: Optional[str]
    version: Optional[str]
    features: List[str]
    feature_count: int

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    training_date: Optional[str] = None
