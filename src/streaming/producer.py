"""
Kafka Sensor Producer
=====================
Simulates 50 IoT sensors continuously publishing readings to Kafka.
Supports configurable anomaly injection for end-to-end pipeline testing.

Usage (standalone):
    python -m src.streaming.producer
"""

import json
import time
from typing import List, Optional

import numpy as np

try:
    from kafka import KafkaProducer as _KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.streaming.kafka_config import (
    BOOTSTRAP_SERVERS,
    PRODUCER_CONFIG,
    SENSOR_TOPIC,
)


class SensorProducer:
    """Publishes IoT sensor readings to the Kafka sensor topic."""

    def __init__(
        self,
        bootstrap_servers: str = BOOTSTRAP_SERVERS,
        topic: str = SENSOR_TOPIC,
    ) -> None:
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed. Run: pip install kafka-python")

        self.topic = topic
        self.producer = _KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            **PRODUCER_CONFIG,
        )
        print(f"  SensorProducer connected → {bootstrap_servers} / {topic}")

    def send_reading(
        self,
        sensor_id: int,
        values: List[float],
        timestamp: Optional[float] = None,
    ) -> None:
        """Publish one reading for a single sensor."""
        message = {
            "timestamp": timestamp or time.time(),
            "sensor_id": sensor_id,
            "values":    values,
        }
        self.producer.send(
            self.topic,
            value=message,
            key=sensor_id,
        )

    def simulate_sensors(
        self,
        n_sensors: int = 50,
        interval_s: float = 1.0,
        inject_anomaly_every: int = 200,
        max_steps: Optional[int] = None,
    ) -> None:
        """Continuously publish synthetic sensor readings.

        Args:
            n_sensors: Number of concurrent sensors to simulate.
            interval_s: Seconds between each full scan of all sensors.
            inject_anomaly_every: Inject anomaly in one sensor every N steps.
            max_steps: Stop after this many steps (None = run forever).
        """
        from src.data.generator import IoTDataGenerator

        gen = IoTDataGenerator(n_sensors=n_sensors)
        step = 0

        print(f"\n  Simulating {n_sensors} sensors "
              f"(anomaly every {inject_anomaly_every} steps)…")
        print("  Press Ctrl+C to stop.\n")

        try:
            while max_steps is None or step < max_steps:
                ts = time.time()
                for sensor_id in range(n_sensors):
                    normal_seq = gen.generate_normal_sequence()
                    # For the producer we send one timestep at a time
                    values = normal_seq[0, :].tolist()

                    if step % inject_anomaly_every == 0 and sensor_id == 0:
                        anomaly_seq = gen.inject_anomaly(normal_seq)
                        values = anomaly_seq[0, :].tolist()
                        print(f"  💉 Injected anomaly at step={step}, sensor={sensor_id}")

                    self.send_reading(sensor_id, values, timestamp=ts)

                self.producer.flush()
                step += 1
                time.sleep(interval_s)

        except KeyboardInterrupt:
            print("\n  Producer stopped.")
        finally:
            self.producer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    producer = SensorProducer()
    producer.simulate_sensors()
