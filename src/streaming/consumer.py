"""
Kafka Anomaly Detector Consumer
================================
Consumes sensor readings from Kafka, maintains a rolling 60-timestep
buffer per sensor, runs the Hybrid Ensemble (Bi-LSTM Autoencoder +
XGBoost) for inference, and publishes rich anomaly alerts to the
alert topic.

Hybrid score = 60 % XGBoost proba + 40 % normalised AE error.

Latency target: < 50 ms per inference sequence.

Usage:
    python -m src.streaming.consumer
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

import joblib
import numpy as np

try:
    from kafka import KafkaConsumer as _KafkaConsumer, KafkaProducer as _KafkaProducer
    from kafka.errors import NoBrokersAvailable, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.features.feature_pipeline import extract_advanced_features
from src.models.trainer import safe_load_keras_model
from src.streaming.kafka_config import (
    ALERT_TOPIC,
    BOOTSTRAP_SERVERS,
    CONSUMER_CONFIG,
    DETECTOR_GROUP,
    PRODUCER_CONFIG,
    SENSOR_TOPIC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Retry helper
# ──────────────────────────────────────────────────────────────────────────────

_KAFKA_RETRY_SECONDS = 10
_KAFKA_MAX_RETRIES   = 12   # ~2 minutes total


def _connect_kafka(
    bootstrap_servers: str,
    input_topic: str,
) -> tuple:
    """Create Kafka consumer + producer with retry on connection failure."""
    for attempt in range(1, _KAFKA_MAX_RETRIES + 1):
        try:
            consumer = _KafkaConsumer(
                input_topic,
                bootstrap_servers=bootstrap_servers,
                group_id=DETECTOR_GROUP,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                **CONSUMER_CONFIG,
            )
            producer = _KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                **PRODUCER_CONFIG,
            )
            logger.info("Kafka connected (attempt %d/%d)", attempt, _KAFKA_MAX_RETRIES)
            return consumer, producer
        except NoBrokersAvailable:
            logger.warning(
                "Kafka not available (attempt %d/%d). Retrying in %ds…",
                attempt, _KAFKA_MAX_RETRIES, _KAFKA_RETRY_SECONDS,
            )
            if attempt == _KAFKA_MAX_RETRIES:
                raise
            time.sleep(_KAFKA_RETRY_SECONDS)


class AnomalyDetectorConsumer:
    """Real-time Hybrid Ensemble anomaly detector via Kafka consumer."""

    # Hybrid weights (must match InferenceEngine)
    _XGB_ALPHA = 0.60
    _AE_ALPHA  = 0.40

    def __init__(
        self,
        input_topic:       str   = SENSOR_TOPIC,
        output_topic:      str   = ALERT_TOPIC,
        bootstrap_servers: str   = BOOTSTRAP_SERVERS,
        seq_length:        int   = 60,
        model_path:        str   = "models/bilstm_autoencoder.h5",
        xgb_path:          str   = "models/xgboost_baseline.json",
        scaler_path:       str   = "models/scaler.pkl",
        threshold_path:    str   = "models/threshold.pkl",
    ) -> None:
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed.")

        self.seq_length   = seq_length
        self.output_topic = output_topic
        self.buffers: dict = defaultdict(list)
        self.n_alerts = 0

        # ── Load Autoencoder (required) ────────────────────────────────
        import tensorflow as tf  # deferred import to speed up cold start
        logger.info("Loading Bi-LSTM Autoencoder from %s…", model_path)
        try:
            self.ae_model  = safe_load_keras_model(model_path)
            self.scaler    = joblib.load(scaler_path)
            self.threshold = float(joblib.load(threshold_path))
            logger.info("Autoencoder ready  threshold=%.6f", self.threshold)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load autoencoder artefacts: {exc}"
            ) from exc

        # Seed AE normalisation EMA
        self._ae_ema = self.threshold
        self._ae_var = (self.threshold * 0.5) ** 2

        # ── Load XGBoost (optional) ────────────────────────────────────
        self.xgb_model = None
        self.mode = "autoencoder"
        if os.path.isfile(xgb_path):
            try:
                import xgboost as xgb
                m = xgb.XGBClassifier()
                m.load_model(xgb_path)
                self.xgb_model = m
                self.mode = "hybrid"
                logger.info("XGBoost loaded from %s → hybrid mode active", xgb_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("XGBoost load failed (%s). AE-only mode.", exc)
        else:
            logger.warning("XGBoost not found at %s. AE-only mode.", xgb_path)

        # ── Kafka connections (with retry) ─────────────────────────────
        self.consumer, self.producer = _connect_kafka(
            bootstrap_servers, input_topic
        )
        logger.info("Consumer ready on topic '%s'", input_topic)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ae_normalised(self, error: float) -> float:
        """On-line EMA-based normalisation of AE error → [0, 1]."""
        alpha = 0.05
        self._ae_ema = alpha * error + (1 - alpha) * self._ae_ema
        self._ae_var = (
            alpha * (error - self._ae_ema) ** 2 + (1 - alpha) * self._ae_var
        )
        std = max(self._ae_var ** 0.5, 1e-8)
        z = (error - self._ae_ema) / std
        return float(np.clip(0.5 + z * 0.15, 0.0, 1.0))

    def _infer(self, sequence: np.ndarray) -> dict:
        """Run Hybrid Ensemble inference on one (T, F) sequence.

        Returns:
            dict with error, xgb_proba, hybrid_score, latency_ms.
        """
        n_feat = sequence.shape[-1]
        scaled = self.scaler.transform(
            sequence.reshape(-1, n_feat)
        ).reshape(1, self.seq_length, n_feat).astype(np.float32)

        t0    = time.perf_counter()
        recon = self.ae_model.predict(scaled, verbose=0)
        error = float(np.mean((scaled - recon) ** 2))
        ae_norm = self._ae_normalised(error)

        xgb_proba: float = -1.0
        if self.mode == "hybrid" and self.xgb_model is not None:
            features  = extract_advanced_features(sequence).reshape(1, -1)
            xgb_proba = float(self.xgb_model.predict_proba(features)[0, 1])
            hybrid_score = self._XGB_ALPHA * xgb_proba + self._AE_ALPHA * ae_norm
        else:
            hybrid_score = ae_norm

        latency_ms = (time.perf_counter() - t0) * 1000

        if latency_ms > 50:
            logger.warning("Inference latency %.1f ms > 50 ms target", latency_ms)

        return {
            "error":        error,
            "xgb_proba":    xgb_proba,
            "hybrid_score": hybrid_score,
            "latency_ms":   latency_ms,
        }

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_message(self, message) -> Optional[dict]:
        """Buffer incoming reads and run inference when window is full.

        Returns:
            Full anomaly_result dict if anomaly detected, else None.
        """
        data      = message.value
        sensor_id = int(data["sensor_id"])
        values    = data["values"]

        self.buffers[sensor_id].append(values)

        if len(self.buffers[sensor_id]) < self.seq_length:
            return None  # Not enough history yet

        # Keep rolling window
        self.buffers[sensor_id] = self.buffers[sensor_id][-self.seq_length:]
        sequence = np.array(self.buffers[sensor_id], dtype=np.float32)

        infer = self._infer(sequence)
        error = infer["error"]

        is_anomaly = error > self.threshold
        severity   = (
            "HIGH"   if error > 2 * self.threshold else
            "MEDIUM" if is_anomaly                  else
            "NONE"
        )

        anomaly_result = {
            "sensor_id":            sensor_id,
            "timestamp":            data.get("timestamp", time.time()),
            "reconstruction_error": round(error, 8),
            "threshold":            self.threshold,
            "is_anomaly":           is_anomaly,
            "severity":             severity,
            "hybrid_score":         round(infer["hybrid_score"], 6),
            "xgb_proba":            round(infer["xgb_proba"], 6),
            "inference_ms":         round(infer["latency_ms"], 2),
            "mode":                 self.mode,
        }

        if is_anomaly:
            self._publish_alert(anomaly_result)

        return anomaly_result if is_anomaly else None

    def _publish_alert(self, anomaly_result: dict) -> None:
        """Send a detailed alert dict to the anomaly_alerts Kafka topic."""
        try:
            self.producer.send(self.output_topic, value=anomaly_result)
            self.n_alerts += 1
            logger.warning(
                "🚨 ANOMALY  sensor=%d  error=%.4f  hybrid=%.4f  severity=%s  "
                "total_alerts=%d",
                anomaly_result["sensor_id"],
                anomaly_result["reconstruction_error"],
                anomaly_result["hybrid_score"],
                anomaly_result["severity"],
                self.n_alerts,
            )
        except KafkaError as exc:
            logger.error("Failed to publish alert to Kafka: %s", exc)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("🟢 Consumer started. Listening for messages…")
        try:
            for message in self.consumer:
                self.process_message(message)
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user.")
        except KafkaError as exc:
            logger.error("Kafka connection lost: %s. Shutting down.", exc)
        finally:
            self.consumer.close()
            self.producer.close()
            logger.info("Kafka connections closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    consumer = AnomalyDetectorConsumer()
    consumer.run()
