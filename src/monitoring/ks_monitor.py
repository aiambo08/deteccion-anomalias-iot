"""
KS Monitor — Kolmogorov-Smirnov Drift Test
============================================
Detects distributional drift using the two-sample KS test.
Complements PSI: KS is more sensitive to distributional shape changes,
PSI is better at detecting proportional shifts.
"""

from typing import List

import numpy as np
from scipy.stats import ks_2samp


class KSMonitor:
    """Two-sample KS test drift monitor for all sensors."""

    def __init__(
        self,
        X_baseline: np.ndarray,
        alpha: float = 0.05,
    ) -> None:
        """
        Args:
            X_baseline: (N, T, F) or (N*T, F) baseline (training) data.
            alpha:      Significance level (default 0.05).
        """
        if X_baseline.ndim == 3:
            B, T, F = X_baseline.shape
            self.baseline = X_baseline.reshape(B * T, F)
        else:
            self.baseline = X_baseline
        self.alpha = alpha

    def check_all_sensors(
        self,
        X_production: np.ndarray,
    ) -> List[dict]:
        """Run KS test for each sensor.

        Args:
            X_production: (N, T, F) or (N*T, F) production data.

        Returns:
            List of dicts for sensors where drift is detected.
            Each dict: {sensor, statistic, p_value, severity}
        """
        if X_production.ndim == 3:
            N, T, F = X_production.shape
            X_prod = X_production.reshape(N * T, F)
        else:
            X_prod = X_production

        n_sensors = X_prod.shape[1]
        drifted = []

        for sensor_idx in range(n_sensors):
            stat, p_value = ks_2samp(
                self.baseline[:, sensor_idx],
                X_prod[:, sensor_idx],
            )
            if p_value < self.alpha:
                severity = "SEVERE" if stat > 0.20 else "MODERATE"
                drifted.append({
                    "sensor":    sensor_idx,
                    "statistic": float(stat),
                    "p_value":   float(p_value),
                    "severity":  severity,
                })

        if drifted:
            severe   = sum(1 for d in drifted if d["severity"] == "SEVERE")
            moderate = sum(1 for d in drifted if d["severity"] == "MODERATE")
            print(f"\n⚠️  KS drift: {len(drifted)} sensors  "
                  f"(severe={severe}, moderate={moderate})")
            for d in sorted(drifted, key=lambda x: -x["statistic"])[:10]:
                print(f"    Sensor {d['sensor']:3d}: KS={d['statistic']:.3f}  "
                      f"p={d['p_value']:.4f}  {d['severity']}")
        else:
            print("✅ KS test: no drift detected across all sensors")

        return drifted
