"""
Kafka Configuration
===================
Centralised topic definitions, broker settings, and message schemas.
"""

# ── Broker ───────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = "localhost:9092"

# ── Topics ───────────────────────────────────────────────────────────────────
SENSOR_TOPIC = "iot_sensor_data"      # Partitions: 3, retention: 7 days
ALERT_TOPIC  = "anomaly_alerts"       # Partitions: 1, retention: 30 days

# ── Consumer groups ───────────────────────────────────────────────────────────
DETECTOR_GROUP = "anomaly_detector_group"

# ── Producer tuning ───────────────────────────────────────────────────────────
PRODUCER_CONFIG = {
    "acks": "all",          # Wait for all ISR replicas
    "retries": 3,
    "batch_size": 16_384,
    "linger_ms": 5,
}

# ── Consumer tuning ───────────────────────────────────────────────────────────
CONSUMER_CONFIG = {
    "auto_offset_reset": "latest",
    "enable_auto_commit": True,
    "max_poll_interval_ms": 300_000,
}

# ── Message schema (reference only — enforced in producer/consumer) ─────────
# Sensor message:
# {
#   "timestamp"  : float  (Unix timestamp),
#   "sensor_id"  : int,
#   "values"     : List[float]  (current timestep readings, length = n_sensors)
# }
#
# Alert message:
# {
#   "sensor_id"          : int,
#   "timestamp"          : float,
#   "reconstruction_error": float,
#   "threshold"          : float,
#   "severity"           : "HIGH" | "MEDIUM"
# }
