"""
FastAPI Pydantic Schemas
========================
Request/response models for the anomaly detection API.
"""

from typing import List, Optional
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
    severity:             str   # "NONE" | "MEDIUM" | "HIGH"
    inference_ms:         Optional[float] = None


class HealthResponse(BaseModel):
    status:    str
    model:     str
    threshold: float
    version:   str = "1.0.0"


class AlertSummary(BaseModel):
    alerts: List[dict]
    total:  int


class MetricsResponse(BaseModel):
    total_alerts:  int
    high_severity: int
    med_severity:  int
    threshold:     float
