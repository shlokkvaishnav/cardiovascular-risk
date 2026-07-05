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
import pandas as pd

# Import consolidated schemas
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
)
from sklearn.ensemble import StackingClassifier

from ..evaluation.explainer import SHAPExplainer, WeightedContributionExplainer
from ..features.derived import (
    DERIVED_CATEGORICAL_FEATURES,
    DERIVED_NUMERICAL_FEATURES,
    compute_derived_features,
)
from ..utils.model_introspection import unwrap_calibrated
from . import auth as auth_router_module
from . import reports as reports_router_module
from . import document_extraction as document_extraction_router_module
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Model-facing feature order (must match preprocessing.numerical_features +
# categorical_features in config.yaml, and the columns produced by
# DataLoader._preprocess_cardio_lifestyle). "bmi" plus the other derived
# clinical features (pulse_pressure, bp_category, bmi_category, age_bucket,
# health_risk_score, bmi_bp_interaction) are computed server-side from the
# client-facing raw fields in PredictionRequest via
# features.derived.compute_derived_features -- the same function DataLoader
# uses at training time -- rather than accepted directly from the client.
FEATURE_NAMES: List[str] = (
    [
        "age",
        "sex",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
    ]
    + DERIVED_NUMERICAL_FEATURES
    + [
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]
    + DERIVED_CATEGORICAL_FEATURES
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
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

# --- Configuration ---
API_KEY_NAME = "X-API-Key"
APP_ENV = os.getenv("APP_ENV", "development").lower()
API_KEY = os.getenv("API_KEY")
if API_KEY is None and APP_ENV in {"development", "dev", "test"}:
    API_KEY = "dev-api-key"
    logger.warning("API_KEY is not set; using a default dev key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
_rate_limit_buckets: Dict[str, List[float]] = {}
_reload_lock = asyncio.Lock()

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

# Optional accounts/report-history feature (Postgres-backed). Entirely
# separate from the core prediction flow -- guest/anonymous usage never
# touches these routes.
app.include_router(auth_router_module.router)
app.include_router(reports_router_module.router)
app.include_router(document_extraction_router_module.router)

# --- Monitoring ---
# Standard HTTP request metrics (latency, request count by path/status) at
# /metrics, plus custom ML-specific metrics below: prediction volume by risk
# level and a distribution of predicted probabilities, both cheap proxies for
# spotting model/data drift over time without a full drift-detection service.
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)

PREDICTIONS_BY_RISK_LEVEL = Counter(
    "cardio_predictions_total",
    "Total predictions served, by risk level",
    ["risk_level"],
)
PREDICTED_PROBABILITY = Histogram(
    "cardio_predicted_probability",
    "Distribution of predicted cardiovascular risk probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Global State
model = None
explainer: Optional[Any] = None  # SHAPExplainer or WeightedContributionExplainer
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
MODEL_PATH = BACKEND_DIR / "models" / "artifacts" / "best_model.pkl"
METADATA_PATH = BACKEND_DIR / "models" / "artifacts" / "training_metadata.json"
SHAP_BACKGROUND_PATH = BACKEND_DIR / "models" / "artifacts" / "shap_background.pkl"
model_metadata = {
    "loaded": False,
    "version": "unknown",
    "path": str(MODEL_PATH),
}


def _to_feature_vector(req: PredictionRequest) -> pd.DataFrame:
    """Build a single-row DataFrame with named columns. The trained Pipeline's
    ColumnTransformer selects columns by name (fit on a DataFrame during
    training), so a raw ndarray fails at transform time with
    'Specifying the columns using strings is only supported for dataframes.'

    BMI and the other derived clinical features (pulse pressure, MAP, BP/BMI
    category, age bucket) are computed here via the same
    features.derived.compute_derived_features used at training time, so
    serving can never drift from what the model was fit on.
    """
    values: Dict[str, float] = req.model_dump()
    row = pd.DataFrame([values])
    row = compute_derived_features(row, feature_engineering_enabled=True)
    return row[FEATURE_NAMES]


def _risk_level(probability: float) -> str:
    return "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"


def _model_predict(
    model_ref, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(model_ref.predict(features), dtype=int)
    if hasattr(model_ref, "predict_proba"):
        probs = np.asarray(model_ref.predict_proba(features), dtype=float)
        probability = probs[:, 1]
        confidence = probs.max(axis=1)
    else:
        probability = predictions.astype(float)
        confidence = np.ones_like(probability)
    return predictions, probability, confidence


def _load_explainer(model_ref):
    """Build a real SHAP explainer for the loaded model, using the background
    sample persisted at training time. Never raises: returns None (and logs)
    on any failure, so a missing/corrupt background never blocks startup.

    If the served model is a stacking ensemble (see ModelTrainer._train_stacking),
    an exact SHAP explanation would require the slow, model-agnostic
    KernelExplainer on the meta-learner -- the same cost that already ruled
    out SVM as a served model. WeightedContributionExplainer stays fast by
    combining each base estimator's own fast/exact explainer, weighted by the
    meta-learner's coefficients, instead."""
    if model_ref is None or not SHAP_BACKGROUND_PATH.exists():
        return None
    try:
        background_df = joblib.load(SHAP_BACKGROUND_PATH)
        # Keep as a DataFrame with named columns: the ColumnTransformer was
        # fit on a DataFrame and requires one at transform time too.
        background = background_df[FEATURE_NAMES]
        unwrapped = unwrap_calibrated(model_ref)
        if isinstance(unwrapped, StackingClassifier):
            return WeightedContributionExplainer(unwrapped, background, FEATURE_NAMES)
        return SHAPExplainer(model_ref, background, FEATURE_NAMES)
    except Exception as exc:
        logger.warning(f"Failed to build SHAP explainer: {exc}")
        return None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        details.append(
            {
                "field": ".".join(
                    [str(loc) for loc in err.get("loc", []) if loc != "body"]
                ),
                "message": err.get("msg"),
            }
        )
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

    logger.info(
        f"Request: {request.method} {request.url.path} Status: {response.status_code} Duration: {process_time:.4f}s"
    )

    logging.setLogRecordFactory(old_factory)
    return response


# --- Auth ---
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if API_KEY is None:
        if APP_ENV in {"development", "dev", "test"}:
            logger.warning(
                "API_KEY is not set; authentication disabled in non-production mode"
            )
            return api_key_header
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    if api_key_header == API_KEY:
        # Basic in-memory rate limiting per API key
        now = time.time()
        bucket = _rate_limit_buckets.setdefault(api_key_header, [])
        window_start = now - 60
        while bucket and bucket[0] < window_start:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_RPM:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        bucket.append(now)
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key",
    )


# --- Events ---
@app.on_event("startup")
async def load_model():
    # model_metadata is a module-level dict mutated in place (item
    # assignment) here, never rebound -- no `global` needed for that.
    global model, explainer
    try:
        if API_KEY is None and APP_ENV not in {"development", "dev", "test"}:
            raise RuntimeError("API_KEY must be set in non-development environments")
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_metadata["loaded"] = True
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    metadata = json.load(f)
                model_metadata["version"] = (
                    metadata.get("config", {})
                    .get("project", {})
                    .get("version", "unknown")
                )
                model_metadata["training_metadata"] = metadata
            else:
                model_metadata["version"] = "unknown"
            model_metadata["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("ML Model loaded successfully")

            explainer = _load_explainer(model)
            if explainer is not None:
                logger.info("SHAP explainer ready")
            else:
                logger.warning(
                    "SHAP explainer unavailable; predictions will omit top_contributors"
                )
        else:
            logger.warning(
                f"Model file not found at {MODEL_PATH}. API starting in skeletal mode."
            )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def init_optional_db():
    """Create accounts/report-history tables if a database is reachable.
    This feature is entirely optional -- a missing/unreachable DB (e.g.
    running the API standalone without the `db` compose service) only
    disables /auth and /reports; it never blocks the core prediction API
    from starting."""
    try:
        from ..db.database import engine
        from ..db import models as db_models

        db_models.Base.metadata.create_all(bind=engine)
        logger.info("Accounts/report-history database ready")
    except Exception as e:
        logger.warning(
            f"Accounts/report-history database unavailable ({e}); /auth and /reports will fail until it is"
        )


# --- Endpoints ---
@app.get("/", tags=["system"])
async def root():
    return {
        "message": "Cardiovascular Risk Prediction API",
        "version": app.version,
        "endpoints": [
            "/health",
            "/predict",
            "/batch-predict",
            "/model/info",
            "/model/reload",
        ],
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
        training_metadata=model_metadata.get("training_metadata"),
    )


@app.post("/model/reload")
async def model_reload():
    global model, explainer
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    async with _reload_lock:
        model = joblib.load(MODEL_PATH)
        model_metadata["loaded"] = True
        model_metadata["loaded_at"] = datetime.utcnow().isoformat()
        explainer = _load_explainer(model)
    return {"message": "Model reloaded", "metadata": model_metadata}


@app.post(
    "/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)]
)
async def predict(req: PredictionRequest, request: Request):
    model_ref = model
    if model_ref is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        features = _to_feature_vector(req)
        prediction_arr, probability_arr, confidence_arr = await asyncio.to_thread(
            _model_predict, model_ref, features
        )

        probability = float(probability_arr[0])
        prediction = int(prediction_arr[0])
        confidence = float(confidence_arr[0])

        top_contributors: Optional[List[Dict[str, float]]] = None
        baseline_probability: Optional[float] = None
        if explainer is not None:
            top_contributors, baseline_probability = await asyncio.to_thread(
                explainer.explain, features
            )

        risk_level = _risk_level(probability)
        PREDICTIONS_BY_RISK_LEVEL.labels(risk_level=risk_level).inc()
        PREDICTED_PROBABILITY.observe(probability)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request.state.request_id,
            top_contributors=top_contributors,
            baseline_probability=baseline_probability,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(get_api_key)],
)
async def batch_predict(batch: BatchPredictionRequest):
    model_ref = model
    if model_ref is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        request_id = str(uuid.uuid4())
        valid_instances: List[PredictionRequest] = []
        errors: List[Dict[str, Any]] = []

        for idx, instance in enumerate(batch.instances):
            try:
                valid_instances.append(PredictionRequest.model_validate(instance))
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})

        if not valid_instances:
            return BatchPredictionResponse(
                predictions=[],
                errors=errors or [{"error": "No valid instances"}],
                total=0,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
            )

        matrix = pd.concat(
            [_to_feature_vector(instance) for instance in valid_instances],
            ignore_index=True,
        )
        predictions, probabilities, confidences = await asyncio.to_thread(
            _model_predict, model_ref, matrix
        )

        # Per-row SHAP explanations are intentionally skipped for batch requests:
        # TreeExplainer/LinearExplainer are cheap, but the KernelExplainer fallback
        # (used for non-tree/non-linear models) is O(n) expensive calls to the model
        # per row, which would make a 100-row batch request unacceptably slow.
        # Use /predict for per-row explanations.
        now = datetime.utcnow().isoformat()
        results = []
        for idx in range(len(valid_instances)):
            probability = float(probabilities[idx])
            risk_level = _risk_level(probability)
            PREDICTIONS_BY_RISK_LEVEL.labels(risk_level=risk_level).inc()
            PREDICTED_PROBABILITY.observe(probability)
            results.append(
                PredictionResponse(
                    prediction=int(predictions[idx]),
                    probability=probability,
                    risk_level=risk_level,
                    confidence=float(confidences[idx]),
                    timestamp=now,
                    request_id=request_id,
                    top_contributors=None,
                    baseline_probability=None,
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            errors=errors if errors else None,
            total=len(results),
            timestamp=now,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal batch prediction error")


# --- Static Files ---
static_dir = REPO_ROOT / "apps" / "web" / "out"
if static_dir.exists():
    app.mount("/_next", StaticFiles(directory=static_dir / "_next"), name="next")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
