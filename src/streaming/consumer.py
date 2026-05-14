"""
Kafka Anomaly Detector Consumer
================================
Consumes sensor readings from Kafka, maintains a rolling 60-timestep
buffer per sensor, runs the Bi-LSTM Autoencoder for inference, and
publishes anomaly alerts to the alert topic.

Latency target: < 50ms per inference sequence.

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
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

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


class AnomalyDetectorConsumer:
    """Real-time anomaly detector via Kafka consumer."""

    def __init__(
        self,
        input_topic: str = SENSOR_TOPIC,
        output_topic: str = ALERT_TOPIC,
        bootstrap_servers: str = BOOTSTRAP_SERVERS,
        seq_length: int = 60,
        model_path: str = "models/bilstm_autoencoder.h5",
        scaler_path: str = "models/scaler.pkl",
        threshold_path: str = "models/threshold.pkl",
    ) -> None:
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed.")

        self.seq_length    = seq_length
        self.output_topic  = output_topic
        self.buffers: dict = defaultdict(list)
        self.n_alerts      = 0

        # ── Load model artefacts ────────────────────────────────────────
        import tensorflow as tf  # deferred import to speed up cold start
        logger.info("Loading model from %s…", model_path)
        self.model     = tf.keras.models.load_model(model_path)
        self.scaler    = joblib.load(scaler_path)
        self.threshold = float(joblib.load(threshold_path))
        logger.info("Threshold = %.6f", self.threshold)

        # ── Kafka connections ───────────────────────────────────────────
        self.consumer = _KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            group_id=DETECTOR_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            **CONSUMER_CONFIG,
        )
        self.producer = _KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            **PRODUCER_CONFIG,
        )
        logger.info("Consumer ready on topic '%s'", input_topic)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, sequence: np.ndarray) -> float:
        """Scale and run autoencoder inference on one (T, F) sequence.

        Returns:
            Mean squared reconstruction error (scalar).
        """
        n_feat = sequence.shape[-1]
        scaled = self.scaler.transform(
            sequence.reshape(-1, n_feat)
        ).reshape(1, self.seq_length, n_feat).astype(np.float32)

        t0 = time.perf_counter()
        recon = self.model.predict(scaled, verbose=0)
        latency_ms = (time.perf_counter() - t0) * 1000

        error = float(np.mean((scaled - recon) ** 2))
        if latency_ms > 50:
            logger.warning("Inference latency %.1f ms > 50ms target", latency_ms)
        return error

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_message(self, message) -> Optional[dict]:
        """Buffer incoming reads and run inference when window is full.

        Returns:
            Alert dict if anomaly detected, else None.
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

        error = self._infer(sequence)

        if error > self.threshold:
            severity = "HIGH" if error > 2 * self.threshold else "MEDIUM"
            alert = {
                "sensor_id":            sensor_id,
                "timestamp":            data.get("timestamp", time.time()),
                "reconstruction_error": error,
                "threshold":            self.threshold,
                "severity":             severity,
            }
            self.producer.send(self.output_topic, value=alert)
            self.n_alerts += 1
            logger.warning(
                "🚨 ANOMALY  sensor=%d  error=%.4f  severity=%s  total_alerts=%d",
                sensor_id, error, severity, self.n_alerts,
            )
            return alert

        return None

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
        finally:
            self.consumer.close()
            self.producer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    consumer = AnomalyDetectorConsumer()
    consumer.run()
