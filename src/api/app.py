from fastapi import FastAPI, HTTPException, status, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime
import os
from pathlib import Path
import time
import uuid

# Import consolidated schemas
from .schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)

# --- Structured Logging Setup ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger("cardio_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# --- Configuration ---
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "cardiovascular-risk-secret-key-123")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- App Initialization ---
app = FastAPI(
    title="Cardiovascular Risk Prediction API",
    description="Production-ready ML API with medical guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model = None
MODEL_PATH = Path("models/artifacts/best_model.pkl")
model_metadata = {
    "loaded": False,
    "version": "unknown",
    "path": str(MODEL_PATH)
}

# --- Middleware ---
@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request_id to logging context temporarily
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    logging.setLogRecordFactory(record_factory)
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(f"Request: {request.method} {request.url.path} Status: {response.status_code} Duration: {process_time:.4f}s")
    
    logging.setLogRecordFactory(old_factory)
    return response

# --- Auth ---
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key"
    )

# --- Events ---
@app.on_event("startup")
async def load_model():
    global model, model_metadata
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_metadata["loaded"] = True
            model_metadata["version"] = "1.0.0" # Ideally read from metadata file
            model_metadata["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("ML Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. API starting in skeletal mode.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_metadata["loaded"] else "degraded",
        model_loaded=model_metadata["loaded"],
        version=model_metadata["version"],
        timestamp=datetime.utcnow().isoformat() 
    )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict(req: PredictionRequest, request: Request):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # Convert Request to Feature Vector (Ensure correct order)
        features = np.array([[
            req.age, req.sex, req.cp, req.trestbps, req.chol, req.fbs,
            req.restecg, req.thalach, req.exang, req.oldpeak, req.slope,
            req.ca, req.thal
        ]])
        
        # Prediction Logic
        # Note: If the model expects specific scaling, it should be part of a Pipeline object in the pkl file.
        # Assuming the pkl contains a full Pipeline including scaler.
        
        prediction = int(model.predict(features)[0])
        
        probability = 0.0
        confidence = 0.0
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            probability = float(probs[1]) # Probability of class 1 (Disease)
            confidence = float(np.max(probs))
        else:
            probability = float(prediction)
            confidence = 1.0 # Fallback for non-probabilistic models

        # Risk Logic
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request.state.request_id
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/batch-predict", response_model=BatchPredictionResponse, dependencies=[Depends(get_api_key)])
async def batch_predict(batch: BatchPredictionRequest):
    if not model:
       raise HTTPException(status_code=503, detail="Model is not loaded")

    results = []
    
    # Logic to optimize batching could go here (e.g. model.predict(batch))
    # For safety, iterating
    for instance in batch.instances:
        # Re-using single predict logic would require internal refactor to avoid HTTP overhead
        # Simple implementation for now:
        try:
             features = np.array([[
                instance.age, instance.sex, instance.cp, instance.trestbps, 
                instance.chol, instance.fbs, instance.restecg, instance.thalach, 
                instance.exang, instance.oldpeak, instance.slope, instance.ca, 
                instance.thal
            ]])
             pred = int(model.predict(features)[0])
             
             probs = [0,0]
             if hasattr(model, "predict_proba"):
                  probs = model.predict_proba(features)[0]
             
             prob = float(probs[1]) if hasattr(model, "predict_proba") else float(pred)
             conf = float(np.max(probs)) if hasattr(model, "predict_proba") else 1.0
             risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
             
             results.append(PredictionResponse(
                 prediction=pred,
                 probability=prob,
                 risk_level=risk,
                 confidence=conf,
                 timestamp=datetime.utcnow().isoformat(),
                 request_id="batch"
             ))
        except Exception as e:
            logger.error(f"Batch item failed: {str(e)}")
            # Continue processing others? Or fail batch? 
            # Production choice: Skip and log.
            continue
            
    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        timestamp=datetime.utcnow().isoformat()
    )

# --- Static Files ---
static_dir = Path("web/out")
if static_dir.exists():
    app.mount("/_next", StaticFiles(directory=static_dir / "_next"), name="next")
    @app.get("/")
    async def root():
        return FileResponse(static_dir / "index.html")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

