"""
FastAPI Application
===================
REST API for real-time IoT anomaly detection.

Endpoints:
  POST /predict          → Hybrid Ensemble inference on a sensor sequence
  GET  /health           → Service health check
  GET  /alerts/recent    → Last N detected anomalies (persisted in SQLite)
  GET  /metrics          → Aggregate alert statistics
  GET  /drift/status     → Latest PSI + KS drift report
  DELETE /alerts         → Clear the alert store (admin use)
  GET  /docs             → Swagger UI (auto-generated)

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import glob
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceEngine
from src.api.schemas import (
    AlertSummary,
    AnomalyResult,
    DriftStatusResponse,
    HealthResponse,
    MetricsResponse,
    SensorSequence,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SQLite Alert Store
# ──────────────────────────────────────────────────────────────────────────────

_DB_PATH    = Path(os.environ.get("ALERTS_DB_PATH", "logs/alerts.db"))
_MAX_ALERTS = int(os.environ.get("MAX_ALERTS", "10000"))


def _init_db() -> None:
    """Create alerts table if it does not exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
                sensor_id            INTEGER NOT NULL,
                is_anomaly           INTEGER NOT NULL,
                reconstruction_error REAL,
                threshold            REAL,
                severity             TEXT,
                hybrid_score         REAL,
                xgb_proba            REAL,
                inference_ms         REAL,
                mode                 TEXT,
                raw_json             TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)"
        )
        conn.commit()
    logger.info("Alert SQLite database ready at %s", _DB_PATH)


def _persist_alert(alert: dict) -> None:
    """Insert one alert row into the SQLite store."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO alerts
                (sensor_id, is_anomaly, reconstruction_error, threshold,
                 severity, hybrid_score, xgb_proba, inference_ms, mode, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.get("sensor_id"),
            int(alert.get("is_anomaly", False)),
            alert.get("reconstruction_error"),
            alert.get("threshold"),
            alert.get("severity"),
            alert.get("hybrid_score"),
            alert.get("xgb_proba"),
            alert.get("inference_ms"),
            alert.get("mode"),
            json.dumps(alert),
        ))
        conn.commit()

    # Prune oldest rows when cap is exceeded
    with sqlite3.connect(_DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        if count > _MAX_ALERTS:
            excess = count - _MAX_ALERTS
            conn.execute(
                "DELETE FROM alerts WHERE id IN "
                "(SELECT id FROM alerts ORDER BY id ASC LIMIT ?)",
                (excess,),
            )
            conn.commit()


def _fetch_recent_alerts(limit: int = 50) -> tuple[List[dict], int]:
    """Return (alerts_list, total_count) from the SQLite store."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT raw_json FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    alerts = [json.loads(r["raw_json"]) for r in rows]
    return alerts, total


def _alert_counts() -> dict:
    """Return aggregate statistics from the SQLite store."""
    with sqlite3.connect(_DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        high  = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE severity = 'HIGH'"
        ).fetchone()[0]
        med   = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE severity = 'MEDIUM'"
        ).fetchone()[0]
    return {"total": total, "high": high, "med": med}


def _clear_alerts() -> None:
    """Truncate the alerts table."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM alerts")
        conn.commit()

# ──────────────────────────────────────────────────────────────────────────────
# ── CORS + Admin API Key configuration ────────────────────────────────────────

_ALLOWED_ORIGINS_ENV = os.environ.get("ALLOWED_ORIGINS", "*")
_ALLOWED_ORIGINS: list = (
    ["*"] if _ALLOWED_ORIGINS_ENV == "*"
    else [o.strip() for o in _ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
)

# Admin key for destructive operations (set via env var in production).
# If unset, DELETE /alerts is disabled for safety.
_ADMIN_API_KEY: Optional[str] = os.environ.get("ADMIN_API_KEY") or None

# ── Drift report helpers ────────────────────────────────────────────────────────

# Matches the pattern written by log_drift_report() in src/monitoring/alerts.py:
#   drift_report_{week}_{hhmmss}.json
_DRIFT_REPORT_GLOB = "logs/drift_reports/drift_report_*.json"


def _latest_drift_report() -> Optional[dict]:
    """Load the most recent DriftPipeline JSON report, or None if none exist."""
    files = sorted(glob.glob(_DRIFT_REPORT_GLOB))
    if not files:
        return None
    with open(files[-1]) as fh:
        report = json.load(fh)
    report["report_file"] = str(files[-1])
    return report

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — model loading on startup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts and initialise DB on startup."""
    # ── Persistent alert store ──────────────────────────────────────────
    _init_db()

    # ── Hybrid Ensemble models ──────────────────────────────────────────
    logger.info("Loading model artefacts…")
    model_path     = os.environ.get("MODEL_PATH",     "models/bilstm_autoencoder.h5")
    xgb_path       = os.environ.get("XGB_PATH",       "models/xgboost_baseline.json")
    scaler_path    = os.environ.get("SCALER_PATH",    "models/scaler.pkl")
    threshold_path = os.environ.get("THRESHOLD_PATH", "models/threshold.pkl")

    try:
        InferenceEngine.load(
            model_path=model_path,
            xgb_path=xgb_path,
            scaler_path=scaler_path,
            threshold_path=threshold_path,
        )
        logger.info("Model ready ✓  mode=%s", InferenceEngine._mode)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        logger.warning("API running WITHOUT model. /predict will return 503.")

    yield
    logger.info("API shutting down.")

# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IoT Anomaly Detection API",
    description=(
        "Real-time anomaly detection for industrial IoT sensors using a "
        "Hybrid Ensemble: 60% XGBoost + 40% Bi-LSTM Autoencoder. "
        "Alerts are persisted in a local SQLite database."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
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

    Returns a Hybrid Ensemble result (60% XGBoost + 40% Autoencoder).
    """
    sequence = np.array(data.sequence, dtype=np.float32)

    if sequence.shape != (60, 50):
        raise HTTPException(
            status_code=422,
            detail=f"sequence must be shape (60, 50), got {sequence.shape}"
        )

    try:
        result = InferenceEngine.predict(sequence)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    response = AnomalyResult(sensor_id=data.sensor_id, **result)

    if response.is_anomaly:
        alert = response.model_dump()
        alert["timestamp"] = datetime.utcnow().isoformat()
        _persist_alert(alert)
        logger.warning(
            "ANOMALY detected: sensor=%d  error=%.4f  hybrid=%.4f  severity=%s",
            data.sensor_id,
            result["reconstruction_error"],
            result["hybrid_score"],
            result["severity"],
        )

    return response


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Check service health, model status, and inference mode."""
    return HealthResponse(
        status="ok",
        model="Hybrid_BiLSTM_XGBoost",
        threshold=InferenceEngine._threshold or 0.0,
        mode=InferenceEngine._mode,
    )


@app.get("/alerts/recent", response_model=AlertSummary, tags=["Monitoring"])
async def get_recent_alerts(limit: int = 50):
    """Return the most recently detected anomaly alerts from the SQLite store.

    Args:
        limit: Maximum number of alerts to return (default 50, max 200).
    """
    limit  = min(limit, 200)
    alerts, total = _fetch_recent_alerts(limit)
    return AlertSummary(alerts=alerts, total=total)


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Return aggregate detection statistics from the persistent alert store."""
    counts = _alert_counts()
    return MetricsResponse(
        total_alerts=counts["total"],
        high_severity=counts["high"],
        med_severity=counts["med"],
        threshold=InferenceEngine._threshold or 0.0,
    )


@app.get("/drift/status", response_model=DriftStatusResponse, tags=["Monitoring"])
async def get_drift_status():
    """Return the latest PSI and KS drift metrics from the last DriftPipeline run.

    Reads the most recent ``logs/drift_reports/drift_report_*.json`` file.
    Returns ``available: false`` if no report has been generated yet.
    """
    report = _latest_drift_report()
    if report is None:
        return DriftStatusResponse(available=False)

    return DriftStatusResponse(
        week=report.get("week"),
        timestamp=report.get("timestamp"),
        psi_critical=report.get("psi_critical"),
        ks_drifted=report.get("ks_drifted"),
        retrain=report.get("retrain"),
        psi_results=report.get("psi_results"),
        ks_results=report.get("ks_results"),
        report_file=report.get("report_file"),
        available=True,
    )


@app.delete("/alerts", tags=["Operations"])
async def clear_alerts(x_admin_key: Optional[str] = Header(default=None)):
    """Clear all persisted alerts (admin use only).

    Requires the ``X-Admin-Key`` request header matching the ``ADMIN_API_KEY``
    environment variable.  If ``ADMIN_API_KEY`` is not set the endpoint is
    disabled (returns 503) to prevent accidental data loss.
    """
    if _ADMIN_API_KEY is None:
        raise HTTPException(
            status_code=503,
            detail="DELETE /alerts is disabled: ADMIN_API_KEY environment variable not set.",
        )
    if x_admin_key != _ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Admin-Key header.",
        )
    _clear_alerts()
    return {"cleared": True}
