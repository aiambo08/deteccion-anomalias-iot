"""
Inference Engine
================
Lazy-loading model singleton for the FastAPI application.
Keeps models in memory across requests.
"""

import os
import time
from typing import Optional

import joblib
import numpy as np


class InferenceEngine:
    """Singleton holding the loaded model and preprocessing artefacts."""

    _model    = None
    _scaler   = None
    _threshold: Optional[float] = None

    @classmethod
    def load(
        cls,
        model_path:     str = "models/bilstm_autoencoder.h5",
        scaler_path:    str = "models/scaler.pkl",
        threshold_path: str = "models/threshold.pkl",
    ) -> None:
        """Load all artefacts into class-level singletons."""
        import tensorflow as tf
        cls._model     = tf.keras.models.load_model(model_path)
        cls._scaler    = joblib.load(scaler_path)
        cls._threshold = float(joblib.load(threshold_path))
        print(f"  InferenceEngine ready  threshold={cls._threshold:.6f}")

    @classmethod
    def predict(cls, sequence: np.ndarray) -> dict:
        """Run full inference pipeline on one (60, 50) sequence.

        Returns dict with error, is_anomaly, severity, and latency_ms.
        """
        if cls._model is None:
            raise RuntimeError("InferenceEngine not loaded. Call InferenceEngine.load() first.")

        n_feat = sequence.shape[-1]
        scaled = cls._scaler.transform(
            sequence.reshape(-1, n_feat)
        ).reshape(1, sequence.shape[0], n_feat).astype(np.float32)

        t0 = time.perf_counter()
        recon = cls._model.predict(scaled, verbose=0)
        latency_ms = (time.perf_counter() - t0) * 1000

        error      = float(np.mean((scaled - recon) ** 2))
        is_anomaly = error > cls._threshold
        severity   = (
            "HIGH"   if error > 2 * cls._threshold else
            "MEDIUM" if is_anomaly                 else
            "NONE"
        )

        return {
            "reconstruction_error": error,
            "threshold":            cls._threshold,
            "is_anomaly":           is_anomaly,
            "severity":             severity,
            "inference_ms":         round(latency_ms, 2),
        }
