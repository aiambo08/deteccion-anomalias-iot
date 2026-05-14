"""
FastAPI Application
===================
REST API for real-time IoT anomaly detection.

Endpoints:
  POST /predict       → Run inference on a sensor sequence
  GET  /health        → Service health check
  GET  /alerts/recent → Last 50 detected anomalies
  GET  /metrics       → Aggregate alert statistics
  GET  /docs          → Swagger UI (auto-generated)

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceEngine
from src.api.schemas import (
    AlertSummary,
    AnomalyResult,
    HealthResponse,
    MetricsResponse,
    SensorSequence,
)

logger = logging.getLogger(__name__)

# In-memory alert store (replace with Redis/DB in production)
_recent_alerts: List[dict] = []
_MAX_ALERTS = 1000  # cap memory usage


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — model loading on startup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts on startup, release on shutdown."""
    logger.info("Loading model artefacts…")
    try:
        InferenceEngine.load()
        logger.info("Model ready ✓")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.warning("API running WITHOUT model. /predict will return 503.")
    yield
    logger.info("API shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IoT Anomaly Detection API",
    description=(
        "Real-time anomaly detection for industrial IoT sensors using "
        "a Bi-LSTM Autoencoder trained exclusively on normal behaviour."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=AnomalyResult, tags=["Inference"])
async def predict_anomaly(data: SensorSequence):
    """Detect anomalies in a multivariate sensor sequence.

    - **sensor_id**: Integer ID of the originating sensor
    - **sequence**: List of 60 timesteps, each with 50 feature values
    """
    sequence = np.array(data.sequence, dtype=np.float32)

    if sequence.shape != (60, 50):
        raise HTTPException(
            status_code=422,
            detail=f"sequence must be shape (60, 50), got {sequence.shape}"
        )

    try:
        result = InferenceEngine.predict(sequence)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = AnomalyResult(
        sensor_id=data.sensor_id,
        **result,
    )

    if response.is_anomaly:
        alert = response.model_dump()
        _recent_alerts.append(alert)
        if len(_recent_alerts) > _MAX_ALERTS:
            _recent_alerts.pop(0)  # drop oldest
        logger.warning(
            "ANOMALY detected: sensor=%d  error=%.4f  severity=%s",
            data.sensor_id,
            result["reconstruction_error"],
            result["severity"],
        )

    return response


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Check service health and model status."""
    return HealthResponse(
        status="ok",
        model="BiLSTM_Autoencoder",
        threshold=InferenceEngine._threshold or 0.0,
    )


@app.get("/alerts/recent", response_model=AlertSummary, tags=["Monitoring"])
async def get_recent_alerts(limit: int = 50):
    """Return the most recently detected anomaly alerts.

    Args:
        limit: Maximum number of alerts to return (default 50, max 200).
    """
    limit = min(limit, 200)
    return AlertSummary(
        alerts=_recent_alerts[-limit:],
        total=len(_recent_alerts),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Return aggregate detection statistics."""
    return MetricsResponse(
        total_alerts=len(_recent_alerts),
        high_severity=sum(1 for a in _recent_alerts if a.get("severity") == "HIGH"),
        med_severity=sum(1 for a in _recent_alerts if a.get("severity") == "MEDIUM"),
        threshold=InferenceEngine._threshold or 0.0,
    )


@app.delete("/alerts", tags=["Operations"])
async def clear_alerts():
    """Clear the in-memory alert store (admin use)."""
    _recent_alerts.clear()
    return {"cleared": True}
