"""
PSI Monitor — Population Stability Index
=========================================
Detects data drift by measuring how much the distribution of production
sensor data has shifted versus the training baseline.

PSI interpretation:
  PSI < 0.10  → No drift — model stable
  PSI 0.10-0.25 → Moderate drift — monitor closely
  PSI ≥ 0.25  → Severe drift — trigger retraining
"""

import os
from datetime import datetime
from typing import List, Tuple

import numpy as np


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two 1-D distributions.

    Args:
        expected: Baseline distribution (training).
        actual:   New distribution (production).
        bins:     Number of histogram bins.

    Returns:
        PSI score (non-negative scalar).
    """
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    # Add small epsilon to handle same-min/max edge case
    bin_edges = np.linspace(min_val - 1e-8, max_val + 1e-8, bins + 1)

    exp_hist, _ = np.histogram(expected, bins=bin_edges)
    act_hist, _ = np.histogram(actual,   bins=bin_edges)

    # Avoid zero frequencies
    exp_pct = np.where(exp_hist == 0, 1e-4, exp_hist / len(expected))
    act_pct = np.where(act_hist == 0, 1e-4, act_hist / len(actual))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


class PSIMonitor:
    """Population Stability Index monitor for all sensors."""

    STATUS_OK       = "OK"
    STATUS_WARNING  = "WARNING"   # PSI 0.10-0.25
    STATUS_CRITICAL = "CRITICAL"  # PSI ≥ 0.25

    def __init__(
        self,
        X_baseline: np.ndarray,
        psi_threshold: float = 0.25,
    ) -> None:
        """
        Args:
            X_baseline: (N, T, F) or (N*T, F) baseline array from training.
            psi_threshold: PSI value above which drift is considered critical.
        """
        # Flatten to (N*T, F) for column-wise distribution comparison
        if X_baseline.ndim == 3:
            B, T, F = X_baseline.shape
            self.baseline = X_baseline.reshape(B * T, F)
        else:
            self.baseline = X_baseline
        self.threshold = psi_threshold

    def _status(self, psi: float) -> str:
        if psi >= self.threshold:
            return self.STATUS_CRITICAL
        if psi >= 0.10:
            return self.STATUS_WARNING
        return self.STATUS_OK

    def check_all_sensors(
        self,
        X_production: np.ndarray,
    ) -> Tuple[dict, List[Tuple[int, float]]]:
        """Run PSI check for all sensors.

        Args:
            X_production: (N, T, F) or (N*T, F) production data.

        Returns:
            results:  Dict sensor_i → {psi, status}
            critical: List of (sensor_idx, psi) for critical sensors.
        """
        if X_production.ndim == 3:
            N, T, F = X_production.shape
            X_prod = X_production.reshape(N * T, F)
        else:
            X_prod = X_production

        n_sensors = X_prod.shape[1]
        results  = {}
        critical = []

        for sensor_idx in range(n_sensors):
            psi = calculate_psi(
                self.baseline[:, sensor_idx],
                X_prod[:, sensor_idx],
            )
            status = self._status(psi)
            results[f"sensor_{sensor_idx}"] = {"psi": psi, "status": status}
            if status == self.STATUS_CRITICAL:
                critical.append((sensor_idx, psi))

        if critical:
            print(f"\n🚨 PSI DRIFT DETECTED — {len(critical)} sensors critical:")
            for idx, psi in sorted(critical, key=lambda x: -x[1]):
                print(f"    Sensor {idx:3d}: PSI={psi:.4f} → {self.STATUS_CRITICAL}")
        else:
            ok_count   = sum(1 for v in results.values() if v["status"] == self.STATUS_OK)
            warn_count = sum(1 for v in results.values() if v["status"] == self.STATUS_WARNING)
            print(f"✅ PSI check complete: {ok_count} OK, {warn_count} warning, {len(critical)} critical")

        return results, critical
