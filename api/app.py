"""
FastAPI Serving Endpoint for Network Attack Onset Forecasting.

Team: TECH ROOKIES, PSC26153 (II-year Engineering Students)
Target: Python 3.10.12, FastAPI, Uvicorn, scikit-learn, joblib.

Endpoints:
- GET /health: Health check endpoint returning {"status": "ok"}
- POST /predict: Predicts onset risk from canonical window features.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Union

from fastapi import Body, FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(
    title="Attack Onset Forecast API",
    description="Containerised FastAPI serving endpoint for network attack onset forecasting.",
    version="1.0.0",
)

# Global cache for the lazily loaded model bundle
_model_bundle = None


def resolve_model_path() -> Path:
    """
    Determines the path to the model bundle.
    Uses MODEL_PATH environment variable if set.
    Defaults to /app/model/model.joblib when running in Docker,
    or <repo_root>/model/model.joblib when running locally.
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    # Check container environment (/app is the working directory in Dockerfile)
    docker_path = Path("/app/model/model.joblib")
    if Path("/app").is_dir():
        return docker_path

    # Local repository fallback
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "model" / "model.joblib"


def load_bundle_lazily() -> Dict[str, Any]:
    """
    Loads model.joblib lazily and retries loading on each request while it is missing.
    Returns HTTP 503 until the model bundle file exists.
    """
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    model_path = resolve_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model bundle not loaded. Target model file does not exist at '{model_path}'. "
                "Please run poc_harness.py to generate and save the model before calling /predict."
            ),
        )

    try:
        loaded = joblib.load(model_path)
        # Validate that the bundle has required keys
        required_keys = ["model", "feature_names", "threshold", "model_name"]
        for k in required_keys:
            if k not in loaded:
                raise ValueError(f"Corrupt model bundle: missing key '{k}'")
        _model_bundle = loaded
        return _model_bundle
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model bundle from '{model_path}': {exc}",
        ) from exc


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Simple healthcheck endpoint.
    Used by docker-compose and monitoring tools.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict_onset(payload: Union[Dict[str, Any], Any] = Body(...)) -> Dict[str, Any]:
    """
    Forecasts network attack onset risk from a single traffic window.

    Accepts either:
    1. Nested payload: {"features": {"flow_count": ..., ...}}
    2. Flat payload: {"flow_count": ..., ...}

    Strict validations:
    - Features must match bundle feature_names exactly.
    - Missing features return HTTP 400.
    - Non-numeric values, NaN, or infinite values return HTTP 400.
    - Missing values are NEVER silently filled with 0.
    """
    bundle = load_bundle_lazily()
    expected_features = bundle["feature_names"]

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid request body. Expected a JSON object.",
        )

    # Extract features dictionary from nested or flat payload
    if "features" in payload and isinstance(payload["features"], dict):
        feat_dict = payload["features"]
    else:
        feat_dict = payload

    # Validate feature completeness
    missing_features = [f for f in expected_features if f not in feat_dict]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required feature(s): {missing_features}. "
                f"Expected features: {expected_features}"
            ),
        )

    # Validate numeric data types and finite values (no bools, strings, NaNs, or infs)
    ordered_values = []
    for feat in expected_features:
        val = feat_dict[feat]

        # In Python, bool is a subclass of int, so explicitly exclude bools
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Non-numeric value detected for feature '{feat}'. "
                    f"Received type '{type(val).__name__}' with value '{val}'."
                ),
            )

        if math.isnan(val) or math.isinf(val):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for feature '{feat}': NaN or Infinite values are not permitted.",
            )

        ordered_values.append(float(val))

    # Reshape into a 2D matrix for scikit-learn
    X = np.array([ordered_values], dtype=np.float64)

    pipeline = bundle["model"]
    threshold = float(bundle["threshold"])
    model_name = str(bundle["model_name"])

    try:
        # Score the window
        probabilities = pipeline.predict_proba(X)
        risk_score = float(probabilities[0, 1])
        alarm = bool(risk_score >= threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed on model: {exc}",
        ) from exc

    return {
        "risk_score": round(risk_score, 4),
        "alarm": alarm,
        "threshold": threshold,
        "model_name": model_name,
    }
