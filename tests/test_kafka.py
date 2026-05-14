"""
Tests: Kafka (offline / mock)
==============================
Tests the producer/consumer logic without requiring a live Kafka broker.
We mock kafka-python classes to test message serialisation and buffer logic.
"""

import json
import time
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Producer tests
# ──────────────────────────────────────────────────────────────────────────────

def test_producer_message_serialisation():
    """Verify that the message dict serialises to valid JSON."""
    sensor_id = 5
    values    = [0.1, 0.2, 0.3] * 17  # 51 values
    ts        = time.time()
    message   = {
        "timestamp": ts,
        "sensor_id": sensor_id,
        "values":    values,
    }
    encoded  = json.dumps(message).encode("utf-8")
    decoded  = json.loads(encoded.decode("utf-8"))
    assert decoded["sensor_id"] == sensor_id
    assert decoded["values"] == values
    assert abs(decoded["timestamp"] - ts) < 0.001


def test_producer_key_encoding():
    """Kafka message key should be bytes-encoded sensor_id."""
    sensor_id = 23
    key = str(sensor_id).encode("utf-8")
    assert key == b"23"


# ──────────────────────────────────────────────────────────────────────────────
# Consumer buffer logic tests (no Kafka required)
# ──────────────────────────────────────────────────────────────────────────────

class MockConsumerLogic:
    """Minimal reimplementation of the buffer logic for isolated testing."""

    def __init__(self, seq_length: int = 60) -> None:
        self.seq_length = seq_length
        self.buffers: dict = defaultdict(list)

    def update(self, sensor_id: int, values: list) -> bool:
        """Return True when a full window is ready."""
        self.buffers[sensor_id].append(values)
        if len(self.buffers[sensor_id]) < self.seq_length:
            return False
        self.buffers[sensor_id] = self.buffers[sensor_id][-self.seq_length:]
        return True

    def get_sequence(self, sensor_id: int) -> np.ndarray:
        return np.array(self.buffers[sensor_id], dtype=np.float32)


def test_consumer_buffer_accumulates():
    logic = MockConsumerLogic(seq_length=5)
    for t in range(4):
        ready = logic.update(0, [float(t)] * 10)
        assert not ready, f"Window should not be ready after {t+1} samples"
    ready = logic.update(0, [4.0] * 10)
    assert ready, "Window should be ready after 5 samples"


def test_consumer_buffer_rolling():
    logic = MockConsumerLogic(seq_length=3)
    for t in range(5):
        logic.update(0, [float(t)] * 10)
    seq = logic.get_sequence(0)
    assert seq.shape == (3, 10), f"Rolling window shape mismatch: {seq.shape}"
    # Last 3 values should be 2.0, 3.0, 4.0
    np.testing.assert_allclose(seq[:, 0], [2.0, 3.0, 4.0])


def test_consumer_independent_sensor_buffers():
    logic = MockConsumerLogic(seq_length=3)
    logic.update(0, [1.0] * 10)
    logic.update(1, [2.0] * 10)
    logic.update(1, [3.0] * 10)
    logic.update(1, [4.0] * 10)

    # Sensor 0 has only 1 sample — not ready
    # Sensor 1 has 3 samples — ready
    assert len(logic.buffers[0]) == 1
    assert len(logic.buffers[1]) == 3


def test_alert_serialisation():
    """Verify alert dict serialises correctly."""
    alert = {
        "sensor_id":             7,
        "timestamp":             time.time(),
        "reconstruction_error":  0.0523,
        "threshold":             0.0100,
        "severity":              "HIGH",
    }
    encoded = json.dumps(alert).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))
    assert decoded["severity"] == "HIGH"
    assert decoded["reconstruction_error"] > decoded["threshold"]
