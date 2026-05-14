"""
Lag / Autocorrelation Features
================================
Temporal regularity and trend features based on autocorrelation and
rolling statistics.

Per sensor (50 sensors × 7 features = 350 total):
  1. autocorr_lag1  : Temporal regularity at lag 1 (normal: ~0.9, anomaly: ~0.3)
  2. autocorr_lag3  : Regularity at lag 3
  3. autocorr_lag7  : Regularity at lag 7
  4. roll_mean_5    : Recent mean (last 5 timesteps)
  5. roll_mean_20   : Mid-term mean (last 20 timesteps)
  6. roll_std_10    : Recent volatility (anomalies: higher std)
  7. linear_trend   : Slope of linear fit on last 10 timesteps → drift indicator

ANOMALY SENSITIVITY:
  - Drift  → linear_trend ≠ 0 and growing
  - Spike  → roll_std_10 spikes, autocorr_lag1 drops
  - Silence → roll_std_10 ≈ 0, roll_mean asymmetric vs baseline
"""

import numpy as np


def _safe_autocorr(s: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at given lag; returns 0.0 on edge cases."""
    if len(s) <= lag + 1:
        return 0.0
    x = s[:-lag]
    y = s[lag:]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    val = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(val) else val


def compute_lag_features(sequence: np.ndarray) -> np.ndarray:
    """Extract autocorrelation and rolling statistics for a single sequence.

    Args:
        sequence: (T, F) array — T timesteps × F sensors.

    Returns:
        features: (F * 7,) float32 array.
    """
    n_sensors = sequence.shape[1]
    features = np.zeros(n_sensors * 7, dtype=np.float32)

    for i in range(n_sensors):
        s = sequence[:, i]
        base = i * 7

        features[base + 0] = _safe_autocorr(s, 1)
        features[base + 1] = _safe_autocorr(s, 3)
        features[base + 2] = _safe_autocorr(s, 7)

        # Rolling statistics (tail windows)
        tail5  = s[-5:]  if len(s) >= 5  else s
        tail20 = s[-20:] if len(s) >= 20 else s
        tail10 = s[-10:] if len(s) >= 10 else s

        features[base + 3] = float(np.mean(tail5))
        features[base + 4] = float(np.mean(tail20))
        features[base + 5] = float(np.std(tail10))

        # Linear trend slope on last 10 timesteps
        if len(tail10) >= 2:
            x = np.arange(len(tail10), dtype=np.float32)
            slope = float(np.polyfit(x, tail10, 1)[0])
            features[base + 6] = slope if np.isfinite(slope) else 0.0

    return features  # (350,)
