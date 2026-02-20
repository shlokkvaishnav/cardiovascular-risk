from fastapi import FastAPI, HTTPException, status, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Import consolidated schemas
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
)

FEATURE_NAMES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


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
if not logger.handlers:
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
    version="1.1.0",
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
    "path": str(MODEL_PATH),
}


def _to_feature_vector(req: PredictionRequest) -> np.ndarray:
    return np.array([[getattr(req, feature) for feature in FEATURE_NAMES]], dtype=np.float64)


def _risk_level(probability: float) -> str:
    return "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"


def _model_predict(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(model.predict(features), dtype=int)
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(features), dtype=float)
        probability = probs[:, 1]
        confidence = probs.max(axis=1)
    else:
        probability = predictions.astype(float)
        confidence = np.ones_like(probability)
    return predictions, probability, confidence


def _build_explanations(features: np.ndarray) -> Optional[List[Dict[str, float]]]:
    """Return a lightweight local explanation proxy using linear coefficients when available."""
    if not hasattr(model, "coef_"):
        return None

    coef = np.asarray(getattr(model, "coef_"), dtype=float)
    if coef.ndim > 1:
        coef = coef[0]
    if coef.shape[0] != features.shape[1]:
        return None

    contrib = np.abs(features[0] * coef)
    if np.allclose(contrib, 0):
        return None

    top_indices = np.argsort(contrib)[-3:][::-1]
    return [{FEATURE_NAMES[idx]: float(contrib[idx])} for idx in top_indices]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        details.append({
            "field": ".".join([str(loc) for loc in err.get("loc", []) if loc != "body"]),
            "message": err.get("msg"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": details,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# --- Middleware ---
@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

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
    response.headers["X-Process-Time"] = f"{process_time:.6f}"

    logger.info(f"Request: {request.method} {request.url.path} Status: {response.status_code} Duration: {process_time:.4f}s")

    logging.setLogRecordFactory(old_factory)
    return response


# --- Auth ---
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key",
    )


# --- Events ---
@app.on_event("startup")
async def load_model():
    global model, model_metadata
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_metadata["loaded"] = True
            model_metadata["version"] = "1.1.0"
            model_metadata["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("ML Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. API starting in skeletal mode.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


# --- Endpoints ---
@app.get("/", tags=["system"])
async def root():
    return {
        "message": "Cardiovascular Risk Prediction API",
        "version": app.version,
        "endpoints": ["/health", "/predict", "/batch-predict", "/model/info", "/model/reload"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_metadata["loaded"] else "degraded",
        model_loaded=model_metadata["loaded"],
        version=model_metadata["version"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        model_path=model_metadata.get("path"),
        loaded=model_metadata["loaded"],
        loaded_at=model_metadata.get("loaded_at"),
        version=model_metadata.get("version"),
        features=FEATURE_NAMES,
    )


@app.post("/model/reload")
async def model_reload():
    global model
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = joblib.load(MODEL_PATH)
    model_metadata["loaded"] = True
    model_metadata["loaded_at"] = datetime.utcnow().isoformat()
    return {"message": "Model reloaded", "metadata": model_metadata}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict(req: PredictionRequest, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        features = _to_feature_vector(req)
        prediction_arr, probability_arr, confidence_arr = await asyncio.to_thread(_model_predict, features)

        probability = float(probability_arr[0])
        prediction = int(prediction_arr[0])
        confidence = float(confidence_arr[0])

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=_risk_level(probability),
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request.state.request_id,
            top_contributors=_build_explanations(features),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/batch-predict", response_model=BatchPredictionResponse, dependencies=[Depends(get_api_key)])
async def batch_predict(batch: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        matrix = np.array(
            [[getattr(instance, feature) for feature in FEATURE_NAMES] for instance in batch.instances],
            dtype=np.float64,
        )
        predictions, probabilities, confidences = await asyncio.to_thread(_model_predict, matrix)

        now = datetime.utcnow().isoformat()
        results = [
            PredictionResponse(
                prediction=int(predictions[idx]),
                probability=float(probabilities[idx]),
                risk_level=_risk_level(float(probabilities[idx])),
                confidence=float(confidences[idx]),
                timestamp=now,
                request_id="batch",
                top_contributors=None,
            )
            for idx in range(len(batch.instances))
        ]

        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            timestamp=now,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal batch prediction error")


# --- Static Files ---
static_dir = Path("web/out")
if static_dir.exists():
    app.mount("/_next", StaticFiles(directory=static_dir / "_next"), name="next")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
