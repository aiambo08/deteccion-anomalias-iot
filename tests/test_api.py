"""
Tests: FastAPI REST API
=======================
End-to-end tests for the FastAPI application using the HTTPX TestClient.

The ``InferenceEngine`` singleton is **mocked** so tests run without requiring
trained model artefacts on disk.  The SQLite alerts store is created in a
temporary directory for isolation.

Run:
    pytest tests/test_api.py -v
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_predict_response(is_anomaly: bool = False, mode: str = "hybrid"):
    """Return a dict that matches InferenceEngine.predict() output."""
    return {
        "is_anomaly":            is_anomaly,
        "hybrid_score":          0.72 if is_anomaly else 0.12,
        "reconstruction_error":  0.053 if is_anomaly else 0.008,
        "xgb_probability":       0.80 if is_anomaly else 0.05,
        "threshold":             0.025,
        "severity":              "HIGH" if is_anomaly else "NONE",
        "mode":                  mode,
        "latency_ms":            18.4,
    }


def _valid_sequence_payload() -> dict:
    """60-timestep × 50-sensor sequence as nested list."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((60, 50)).tolist()
    return {"sequence": data}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Test client with a fully mocked InferenceEngine and a temporary SQLite DB.

    The lifespan startup is bypassed so no model files are needed on disk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        alerts_db = os.path.join(tmp, "alerts_test.db")

        with (
            patch("src.api.main.InferenceEngine") as mock_engine_cls,
            patch.dict(os.environ, {
                "ALERTS_DB_PATH": alerts_db,
                "ADMIN_API_KEY":  "test-secret-key",
            }),
        ):
            mock_engine = MagicMock()
            mock_engine.predict.return_value = _make_predict_response(False)
            mock_engine_cls.load.return_value = None
            mock_engine_cls.predict.side_effect = mock_engine.predict

            # Patch the module-level singleton methods used in route handlers
            with patch("src.api.main.InferenceEngine.predict",
                       side_effect=mock_engine.predict):

                from src.api.main import app
                with TestClient(app, raise_server_exceptions=True) as c:
                    yield c, mock_engine


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200


def test_health_payload_structure(client):
    c, _ = client
    body = c.get("/health").json()
    assert "status" in body
    assert "uptime_seconds" in body


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_normal_returns_200(client):
    c, mock_engine = client
    mock_engine.predict.return_value = _make_predict_response(False)
    response = c.post("/predict", json=_valid_sequence_payload())
    assert response.status_code == 200


def test_predict_response_fields(client):
    c, mock_engine = client
    mock_engine.predict.return_value = _make_predict_response(False)
    body = c.post("/predict", json=_valid_sequence_payload()).json()
    for field in ("is_anomaly", "hybrid_score", "reconstruction_error",
                  "threshold", "severity", "mode"):
        assert field in body, f"Missing field: {field}"


def test_predict_anomaly_flag(client):
    c, mock_engine = client
    mock_engine.predict.return_value = _make_predict_response(True)
    body = c.post("/predict", json=_valid_sequence_payload()).json()
    assert body["is_anomaly"] is True
    assert body["severity"] in ("HIGH", "MEDIUM")


def test_predict_wrong_shape_returns_422(client):
    """Sequence with wrong shape should be rejected before inference."""
    c, _ = client
    bad_payload = {"sequence": [[0.1] * 10] * 30}   # (30, 10) instead of (60, 50)
    response = c.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_body_returns_422(client):
    c, _ = client
    response = c.post("/predict", json={})
    assert response.status_code == 422


# ── /alerts/recent ────────────────────────────────────────────────────────────

def test_alerts_recent_returns_list(client):
    c, _ = client
    body = c.get("/alerts/recent").json()
    assert isinstance(body, list)


def test_alerts_recent_limit_param(client):
    c, _ = client
    response = c.get("/alerts/recent?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert len(body) <= 5


# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics_returns_200(client):
    c, _ = client
    response = c.get("/metrics")
    assert response.status_code == 200


def test_metrics_payload_fields(client):
    c, _ = client
    body = c.get("/metrics").json()
    assert "total_predictions" in body
    assert "anomaly_rate" in body


# ── /drift/status ─────────────────────────────────────────────────────────────

def test_drift_status_returns_200(client):
    c, _ = client
    response = c.get("/drift/status")
    assert response.status_code == 200


def test_drift_status_has_available_field(client):
    c, _ = client
    body = c.get("/drift/status").json()
    assert "available" in body


# ── DELETE /alerts ────────────────────────────────────────────────────────────

def test_delete_alerts_without_key_returns_401(client):
    """Calling DELETE /alerts without the API key should return 401."""
    c, _ = client
    response = c.delete("/alerts")
    assert response.status_code in (401, 503)


def test_delete_alerts_with_wrong_key_returns_401(client):
    c, _ = client
    response = c.delete("/alerts", headers={"X-Admin-Key": "wrong-key"})
    assert response.status_code == 401


def test_delete_alerts_with_correct_key_returns_200(client):
    c, _ = client
    response = c.delete("/alerts", headers={"X-Admin-Key": "test-secret-key"})
    assert response.status_code == 200
    assert response.json().get("cleared") is True
