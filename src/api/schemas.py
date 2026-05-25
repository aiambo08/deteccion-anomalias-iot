"""
FastAPI Pydantic Schemas
========================
Request/response models for the anomaly detection API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SensorSequence(BaseModel):
    """Input: one multivariate sensor sequence."""
    sensor_id: int = Field(..., description="Unique sensor identifier (0-49)")
    sequence: List[List[float]] = Field(
        ...,
        description="Shape (60, 50) — 60 timesteps × 50 sensor channels"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "sensor_id": 5,
            "sequence": [[0.0] * 50] * 60,
        }
    }}


class AnomalyResult(BaseModel):
    """Response: anomaly detection result for one sequence."""
    sensor_id:            int
    is_anomaly:           bool
    reconstruction_error: float
    threshold:            float
    severity:             str         # "NONE" | "MEDIUM" | "HIGH"
    hybrid_score:         float = 0.0 # combined ensemble score in [0, 1]
    xgb_proba:            float = -1.0  # XGB anomaly probability (−1 if unavailable)
    inference_ms:         Optional[float] = None
    mode:                 str  = "autoencoder"  # "hybrid" | "autoencoder"


class HealthResponse(BaseModel):
    status:     str
    model:      str
    threshold:  float
    mode:       str   = "autoencoder"
    version:    str   = "2.0.0"


class AlertSummary(BaseModel):
    alerts: List[dict]
    total:  int


class MetricsResponse(BaseModel):
    total_alerts:  int
    high_severity: int
    med_severity:  int
    threshold:     float


class DriftStatusResponse(BaseModel):
    """Latest PSI + KS drift metrics from the most recent DriftPipeline run."""
    week:           Optional[str]  = None
    timestamp:      Optional[str]  = None
    psi_critical:   Optional[int]  = None   # number of sensors with PSI ≥ threshold
    ks_drifted:     Optional[int]  = None   # number of KS-drifted sensors
    retrain:        Optional[bool] = None   # was retraining triggered?
    psi_results:    Optional[Dict[str, Any]] = None
    ks_results:     Optional[Dict[str, Any]] = None
    report_file:    Optional[str]  = None
    available:      bool = True
