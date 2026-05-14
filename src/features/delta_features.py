"""
Delta Features
==============
Temporal derivative features that capture velocity and acceleration in sensor signals.

Per sensor (50 sensors × 5 features = 250 total):
  - delta_1     : Instantaneous velocity  (s[t] - s[t-1])
  - delta_3     : Short-term trend        (s[t] - s[t-3])
  - delta_5     : Medium-term trend       (s[t] - s[t-5])
  - accel       : Acceleration            (delta_1[t] - delta_1[t-1])
  - max_abs_delta: Max abs change in window → spike indicator

ANOMALY SENSITIVITY:
  - Spike → delta_1 or max_abs_delta >> normal range
  - Drift  → accel sustained positive/negative
  - Shift  → delta_1 spike at transition point
"""

import numpy as np


def compute_delta_features(sequence: np.ndarray) -> np.ndarray:
    """Extract delta/derivative features from a single sequence.

    Args:
        sequence: (T, F) array — T timesteps × F sensors.

    Returns:
        features: (F * 5,) float32 array.
    """
    n_sensors = sequence.shape[1]
    features = np.zeros(n_sensors * 5, dtype=np.float32)

    for i in range(n_sensors):
        s = sequence[:, i]
        base = i * 5

        diff = np.diff(s)  # length T-1

        features[base + 0] = s[-1] - s[-2] if len(s) > 1 else 0.0   # delta_1
        features[base + 1] = s[-1] - s[-4] if len(s) > 3 else 0.0   # delta_3
        features[base + 2] = s[-1] - s[-6] if len(s) > 5 else 0.0   # delta_5

        # Acceleration = change in velocity
        if len(s) > 2:
            vel_curr = s[-1] - s[-2]
            vel_prev = s[-2] - s[-3]
            features[base + 3] = vel_curr - vel_prev
        # else remains 0.0

        features[base + 4] = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0  # max_abs_delta

    return features  # (250,)
