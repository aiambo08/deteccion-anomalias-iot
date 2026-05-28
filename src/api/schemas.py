"""
FastAPI Pydantic Schemas
========================
Request/response models for the anomaly detection API.

Change log
----------
* sensor_id made optional (default -1) so callers that only POST a raw
  sequence without an explicit sensor_id still receive a valid response.
* /alerts/recent now returns a plain list (AlertList) instead of a
  wrapped object — see test_api.py contract.
* MetricsResponse exposes total_predictions + anomaly_rate in addition to
  severity counts so monitoring dashboards can compute them directly.
* HealthResponse.mode defaults to empty string to survive test mocks that
  patch InferenceEngine as a MagicMock (Pydantic must receive a str).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SensorSequence(BaseModel):
    """Input: one multivariate sensor sequence.

    ``sensor_id`` is optional — if omitted it defaults to -1 (anonymous).
    """
    sensor_id: int = Field(default=-1, description="Unique sensor identifier (0-49)")
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
    sensor_id:            int     = -1
    is_anomaly:           bool
    reconstruction_error: float
    threshold:            float
    severity:             str                # "NONE" | "MEDIUM" | "HIGH"
    hybrid_score:         float  = 0.0       # combined ensemble score in [0, 1]
    xgb_proba:            float  = -1.0      # XGB anomaly probability (−1 if unavailable)
    inference_ms:         Optional[float] = None
    mode:                 str    = "autoencoder"  # "hybrid" | "autoencoder"


class HealthResponse(BaseModel):
    status:          str
    model:           str
    threshold:       float
    mode:            str   = ""    # default empty string — safe when engine is not loaded
    version:         str   = "2.0.0"
    uptime_seconds:  float = 0.0   # seconds since API startup


# /alerts/recent returns a plain JSON list — no wrapper object.
# Use List[dict] as the response_model so FastAPI serialises correctly.
AlertList = List[dict]


class AlertSummary(BaseModel):
    """Internal wrapper; not used as response_model for /alerts/recent."""
    alerts: List[dict]
    total:  int


class MetricsResponse(BaseModel):
    """Aggregate detection statistics.

    Includes both raw severity counts and the derived metrics
    (total_predictions, anomaly_rate) expected by monitoring clients.
    """
    total_alerts:      int
    total_predictions: int        # same as total_alerts — every prediction is logged
    high_severity:     int
    med_severity:      int
    threshold:         float
    anomaly_rate:      float      # fraction of predictions that were anomalies


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
