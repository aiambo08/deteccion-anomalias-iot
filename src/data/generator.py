"""
IoT Data Generator
==================
Generates synthetic multi-sensor IoT data with realistic signal patterns
and configurable anomaly injection. Produces labelled (N, 60, 50) arrays.

Signal types:
  - Vibration  (sensors 0, 4, 8, ...)  : sinusoidal + Gaussian noise
  - Temperature(sensors 1, 5, 9, ...)  : slow ramp + soft noise
  - Pressure   (sensors 2, 6, 10, ...) : stationary oscillation
  - RPM        (sensors 3, 7, 11, ...) : quasi-constant with small fluctuation

Anomaly types: spike, drift, shift, periodic, silence
"""

import numpy as np
from typing import Tuple


class IoTDataGenerator:
    """Synthetic IoT multi-sensor data generator with anomaly injection."""

    ANOMALY_TYPES = ["spike", "drift", "shift", "periodic", "silence"]

    def __init__(
        self,
        n_sensors: int = 50,
        seq_length: int = 60,
        n_sequences: int = 40_000,
        anomaly_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_sensors = n_sensors
        self.seq_length = seq_length
        self.n_sequences = n_sequences
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Normal signals
    # ------------------------------------------------------------------

    def _vibration(self) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, self.seq_length)
        return np.sin(10 * t) + np.random.normal(0, 0.03, self.seq_length)

    def _temperature(self) -> np.ndarray:
        base = np.random.normal(50, 1)
        return np.linspace(base, base + 2, self.seq_length) + np.random.normal(
            0, 0.5, self.seq_length
        )

    def _pressure(self) -> np.ndarray:
        return 100.0 + np.sin(
            np.linspace(0, np.pi, self.seq_length)
        ) + np.random.normal(0, 2, self.seq_length)

    def _rpm(self) -> np.ndarray:
        return np.random.normal(600, 10, self.seq_length)

    _SIGNAL_MAP = {0: "_vibration", 1: "_temperature", 2: "_pressure", 3: "_rpm"}

    def generate_normal_sequence(self) -> np.ndarray:
        """Return one normal multivariate sequence of shape (seq_length, n_sensors)."""
        sequence = np.zeros((self.seq_length, self.n_sensors))
        for s in range(self.n_sensors):
            method = self._SIGNAL_MAP[s % 4]
            sequence[:, s] = getattr(self, method)()
        return sequence

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------

    def inject_anomaly(
        self, sequence: np.ndarray, anomaly_type: str | None = None
    ) -> np.ndarray:
        """Inject a single anomaly into a copy of the sequence.

        Args:
            sequence: (seq_length, n_sensors) normal sequence.
            anomaly_type: One of ANOMALY_TYPES, or None for random.

        Returns:
            Modified sequence with one anomaly injected.
        """
        seq = sequence.copy()
        if anomaly_type is None:
            anomaly_type = np.random.choice(self.ANOMALY_TYPES)

        sensor_idx = np.random.randint(0, self.n_sensors)
        s = seq[:, sensor_idx]
        sigma = np.std(s) if np.std(s) > 0 else 1.0

        if anomaly_type == "spike":
            t_idx = np.random.randint(5, self.seq_length - 5)
            magnitude = np.random.choice([-1.0, 1.0]) * np.random.uniform(4.0, 6.0)
            seq[t_idx, sensor_idx] += magnitude * sigma

        elif anomaly_type == "drift":
            start = np.random.randint(20, 40)
            slope = np.random.uniform(0.05, 0.15)
            length = self.seq_length - start
            seq[start:, sensor_idx] += np.arange(length) * slope

        elif anomaly_type == "shift":
            start = np.random.randint(15, 30)
            shift_val = np.random.uniform(3.0, 6.0) * sigma
            seq[start:, sensor_idx] += shift_val

        elif anomaly_type == "periodic":
            t = np.linspace(0, 4 * np.pi, self.seq_length)
            # Inject a new harmonic at 1.5× the nominal vibration frequency
            freq_anomaly = 1.5 * 10
            seq[:, sensor_idx] += 0.3 * np.sin(freq_anomaly * t)

        elif anomaly_type == "silence":
            start = np.random.randint(20, 40)
            end = min(start + np.random.randint(8, 15), self.seq_length)
            seq[start:end, sensor_idx] = 0.0

        return seq

    # ------------------------------------------------------------------
    # Full dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the complete labelled dataset.

        Returns:
            X: (n_sequences, seq_length, n_sensors) float32 array.
            y: (n_sequences,) int32 array — 0=Normal, 1=Anomaly.
        """
        n_anomalies = int(self.n_sequences * self.anomaly_ratio)
        n_normal = self.n_sequences - n_anomalies

        X_list, y_list = [], []

        for _ in range(n_normal):
            X_list.append(self.generate_normal_sequence())
            y_list.append(0)

        for _ in range(n_anomalies):
            seq = self.generate_normal_sequence()
            seq = self.inject_anomaly(seq)
            X_list.append(seq)
            y_list.append(1)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Shuffle preserving index alignment
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    # ------------------------------------------------------------------
    # Convenience CLI entry point
    # ------------------------------------------------------------------

    def run_and_save(self, output_dir: str = "data/raw") -> None:
        """Generate full dataset and save as .npy files."""
        import os

        print(f"Generating {self.n_sequences:,} sequences "
              f"({self.n_sensors} sensors, {self.seq_length} timesteps)…")
        X, y = self.generate_dataset()
        print(f"  X shape : {X.shape}   dtype={X.dtype}")
        print(f"  y shape : {y.shape}   anomaly rate={y.mean()*100:.1f}%")

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y.npy"), y)
        print(f"  Saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = IoTDataGenerator()
    gen.run_and_save()
