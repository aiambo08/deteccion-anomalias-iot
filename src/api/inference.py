"""
Inference Engine
================
Lazy-loading model singleton for the FastAPI application.
Keeps both the Bi-LSTM Autoencoder and the XGBoost model in memory
across requests and combines them as a Hybrid Ensemble:

    hybrid_score = 0.60 × XGB_proba + 0.40 × AE_error_normalised

Latency target: < 50 ms per inference sequence.
"""

import logging
import os
import time
from typing import Optional

import joblib
import numpy as np

from src.features.feature_pipeline import extract_advanced_features

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Singleton holding all loaded model and preprocessing artefacts.

    Supports two inference modes:
      - ``hybrid``   : 60 % XGBoost + 40 % Autoencoder (default)
      - ``autoencoder``: Autoencoder-only fallback when XGBoost is absent
    """

    # ── Class-level singletons ──────────────────────────────────────────
    _ae_model   = None          # Keras Bi-LSTM Autoencoder
    _xgb_model  = None          # XGBClassifier (optional)
    _scaler     = None          # StandardScaler fitted on raw features
    _threshold: Optional[float] = None  # Autoencoder p95 threshold
    _mode: str  = "autoencoder" # updated to "hybrid" when XGB loads OK

    # Hybrid weights
    _XGB_ALPHA  = 0.60          # 60 % supervised
    _AE_ALPHA   = 0.40          # 40 % unsupervised

    # Running error statistics for on-line normalisation of AE scores
    _ae_error_ema: float = 0.0
    _ae_error_var: float = 1.0
    _ae_ema_alpha: float = 0.05  # EMA smoothing factor

    @classmethod
    def load(
        cls,
        model_path:     str = "models/bilstm_autoencoder.h5",
        xgb_path:       str = "models/xgboost_baseline.json",
        scaler_path:    str = "models/scaler.pkl",
        threshold_path: str = "models/threshold.pkl",
    ) -> None:
        """Load all artefacts into class-level singletons.

        XGBoost is loaded opportunistically — if the file does not exist
        the engine falls back to autoencoder-only mode with a warning.
        """
        import tensorflow as tf  # deferred import for cold-start speed

        # ── Autoencoder (required) ──────────────────────────────────────
        logger.info("Loading Bi-LSTM Autoencoder from %s…", model_path)
        cls._ae_model  = tf.keras.models.load_model(model_path)
        cls._scaler    = joblib.load(scaler_path)
        cls._threshold = float(joblib.load(threshold_path))
        logger.info("Autoencoder ready  threshold=%.6f", cls._threshold)

        # Seed EMA with threshold so first window normalises correctly
        cls._ae_error_ema = cls._threshold
        cls._ae_error_var = (cls._threshold * 0.5) ** 2

        # ── XGBoost (optional but preferred) ───────────────────────────
        if os.path.isfile(xgb_path):
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(xgb_path)
                cls._xgb_model = model
                cls._mode = "hybrid"
                logger.info("XGBoost loaded from %s → hybrid mode active", xgb_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "XGBoost load failed (%s). Falling back to autoencoder-only.", exc
                )
                cls._mode = "autoencoder"
        else:
            logger.warning(
                "XGBoost artefact not found at %s. Running in autoencoder-only mode.",
                xgb_path,
            )
            cls._mode = "autoencoder"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _scale_sequence(cls, sequence: np.ndarray) -> np.ndarray:
        """Scale a (T, F) sequence; returns (1, T, F) float32 tensor."""
        n_feat = sequence.shape[-1]
        scaled = cls._scaler.transform(
            sequence.reshape(-1, n_feat)
        ).reshape(1, sequence.shape[0], n_feat).astype(np.float32)
        return scaled

    @classmethod
    def _ae_error(cls, scaled: np.ndarray) -> float:
        """Run Autoencoder inference and return MSE reconstruction error."""
        recon = cls._ae_model.predict(scaled, verbose=0)
        return float(np.mean((scaled - recon) ** 2))

    @classmethod
    def _ae_normalised(cls, error: float) -> float:
        """Normalise AE error to [0, 1] via on-line EMA statistics.

        Uses a running mean + variance estimate so the normalisation
        adapts without requiring the full error history in memory.
        """
        # Update EMA
        alpha = cls._ae_ema_alpha
        cls._ae_error_ema = alpha * error + (1 - alpha) * cls._ae_error_ema
        cls._ae_error_var = (
            alpha * (error - cls._ae_error_ema) ** 2
            + (1 - alpha) * cls._ae_error_var
        )
        std = max(cls._ae_error_var ** 0.5, 1e-8)
        # Sigmoid-like clamp to [0, 1]
        z = (error - cls._ae_error_ema) / std
        return float(np.clip(0.5 + z * 0.15, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def predict(cls, sequence: np.ndarray) -> dict:
        """Run full Hybrid Ensemble inference on one (60, 50) sequence.

        Returns:
            dict with keys:
              - reconstruction_error (float)
              - threshold            (float)
              - is_anomaly           (bool)
              - severity             (str)  "NONE" | "MEDIUM" | "HIGH"
              - hybrid_score         (float) in [0, 1]
              - xgb_proba            (float) XGB anomaly probability (−1 if unavailable)
              - inference_ms         (float)
              - mode                 (str)  "hybrid" | "autoencoder"
        """
        if cls._ae_model is None:
            raise RuntimeError(
                "InferenceEngine not loaded. Call InferenceEngine.load() first."
            )

        t0     = time.perf_counter()
        scaled = cls._scale_sequence(sequence)

        # ── Autoencoder branch ─────────────────────────────────────────
        error      = cls._ae_error(scaled)
        ae_norm    = cls._ae_normalised(error)

        # ── XGBoost branch (hybrid mode only) ─────────────────────────
        xgb_proba: float = -1.0
        if cls._mode == "hybrid" and cls._xgb_model is not None:
            features   = extract_advanced_features(sequence).reshape(1, -1)
            xgb_proba  = float(cls._xgb_model.predict_proba(features)[0, 1])
            hybrid_score = cls._XGB_ALPHA * xgb_proba + cls._AE_ALPHA * ae_norm
        else:
            hybrid_score = ae_norm  # fallback: use normalised AE score

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Decision ───────────────────────────────────────────────────
        # Hybrid mode: ensemble score is authoritative (> 0.5 = anomaly).
        # Autoencoder-only: AE reconstruction error vs p95 threshold.
        if cls._mode == "hybrid":
            is_anomaly = hybrid_score > 0.5
        else:
            is_anomaly = error > cls._threshold

        severity   = (
            "HIGH"   if error > 2 * cls._threshold else
            "MEDIUM" if is_anomaly                  else
            "NONE"
        )

        if latency_ms > 50:
            logger.warning("Inference latency %.1f ms exceeds 50 ms target", latency_ms)

        return {
            "reconstruction_error": round(error, 8),
            "threshold":            cls._threshold,
            "is_anomaly":           is_anomaly,
            "severity":             severity,
            "hybrid_score":         round(hybrid_score, 6),
            "xgb_proba":            round(xgb_proba, 6),
            "inference_ms":         round(latency_ms, 2),
            "mode":                 cls._mode,
        }
