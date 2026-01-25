from fastapi import FastAPI, HTTPException, status, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import os
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "cardiovascular-risk-secret-key-123") # Default for dev
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API Key"""
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Cardiovascular Risk Prediction API",
    description="Production-ready ML API for predicting cardiovascular risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
model_metadata = {
    "loaded": False,
    "model_path": None,
    "loaded_at": None,
    "version": None
}

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Invalid input data. Please check the request format."
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred"
        }
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their processing time"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Completed in {process_time:.3f}s with status {response.status_code}")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Pydantic models with validation
class PredictionRequest(BaseModel):
    """Request model for cardiovascular risk prediction"""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0: false, 1: true)")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0: no, 1: yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)")
    
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
    probability: float = Field(..., ge=0, le=1, description="Probability of cardiovascular disease")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    confidence: float = Field(..., description="Model confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")
    
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[PredictionRequest] = Field(..., max_items=100, description="List of prediction requests (max 100)")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_metadata: Dict[str, Any]
    timestamp: str
    uptime: Optional[float] = None

class ModelInfo(BaseModel):
    """Model information response"""
    model_path: Optional[str]
    loaded: bool
    loaded_at: Optional[str]
    version: Optional[str]
    features: List[str]

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, model_metadata
    
    model_path = Path("models/artifacts/best_model.pkl")
    
    try:
        if model_path.exists():
            model = joblib.load(model_path)
            model_metadata.update({
                "loaded": True,
                "model_path": str(model_path),
                "loaded_at": datetime.now().isoformat(),
                "version": "1.0.0"
            })
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. API will return 503 for predictions.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        model_metadata["loaded"] = False

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cardiovascular Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    Returns API status and model availability
    """
    return HealthResponse(
        status="healthy" if model_metadata["loaded"] else "degraded",
        model_loaded=model_metadata["loaded"],
        model_metadata=model_metadata,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    return ModelInfo(
        model_path=model_metadata.get("model_path"),
        loaded=model_metadata["loaded"],
        loaded_at=model_metadata.get("loaded_at"),
        version=model_metadata.get("version"),
        features=features
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(get_api_key)])
async def predict(request: PredictionRequest):
    """
    Make a single prediction for cardiovascular risk
    
    - **age**: Age in years (0-120)
    - **sex**: Sex (0: female, 1: male)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure in mm Hg (80-200)
    - **chol**: Serum cholesterol in mg/dl (100-600)
    - **fbs**: Fasting blood sugar > 120 mg/dl (0: false, 1: true)
    - **restecg**: Resting ECG results (0-2)
    - **thalach**: Maximum heart rate achieved (60-220)
    - **exang**: Exercise induced angina (0: no, 1: yes)
    - **oldpeak**: ST depression induced by exercise (0-10)
    - **slope**: Slope of peak exercise ST segment (0-2)
    - **ca**: Number of major vessels colored by fluoroscopy (0-4)
    - **thal**: Thalassemia (0-3)
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare features
        features = np.array([[
            request.age, request.sex, request.cp, request.trestbps,
            request.chol, request.fbs, request.restecg, request.thalach,
            request.exang, request.oldpeak, request.slope, request.ca,
            request.thal
        ]])
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            probability = float(probabilities[1])
            confidence = float(max(probabilities))
        else:
            # For models without predict_proba, use decision function if available
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(features)[0]
                # Convert to probability-like score
                probability = float(1 / (1 + np.exp(-decision)))
                confidence = abs(decision)
            else:
                probability = float(prediction)
                confidence = 1.0
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        logger.info(f"Prediction made: {prediction} (probability: {probability:.3f})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"], dependencies=[Depends(get_api_key)])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple instances
    Maximum 100 instances per request
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    predictions = []
    
    for instance in request.instances:
        try:
            pred_response = await predict(instance)
            predictions.append(pred_response)
        except Exception as e:
            logger.error(f"Batch prediction error for instance: {e}")
            # Continue with other predictions
            continue
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        timestamp=datetime.now().isoformat()
    )

@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model from disk
    Useful for updating the model without restarting the server
    """
    global model, model_metadata
    
    model_path = Path("models/artifacts/best_model.pkl")
    
    try:
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found at {model_path}"
            )
        
        model = joblib.load(model_path)
        model_metadata.update({
            "loaded": True,
            "model_path": str(model_path),
            "loaded_at": datetime.now().isoformat(),
            "version": "1.0.0"
        })
        
        logger.info("Model reloaded successfully")
        
        return {
            "message": "Model reloaded successfully",
            "metadata": model_metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files (Frontend)
# Must be after API routes to avoid conflict
static_dir = Path("web/out")

if static_dir.exists():
    app.mount("/_next", StaticFiles(directory=static_dir / "_next"), name="next")
    
    @app.get("/")
    async def serve_root():
        return FileResponse(static_dir / "index.html")

    # Serve other static files (favicon, etc)
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

elif os.getenv("ENV") == "production":
    logger.warning("Frontend static files not found at web/out")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

