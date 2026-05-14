"""
FFT Features
============
Frequency-domain features that expose spectral signatures of sensor signals.

Per sensor (50 sensors × 9 features = 450 total):
  1. fundamental_power  : Power at the expected fundamental frequency
  2-6. harmonic_power_2..5: Power at harmonics 2×, 3×, 4×, 5×
  7. harmonic_ratio     : Σ(harmonic powers) / fundamental → bearing fault indicator
  8. dominant_freq      : Frequency with maximum spectral power
  9. spectral_entropy   : Shannon entropy of normalised power spectrum → chaos indicator

ANOMALY SENSITIVITY:
  - Bearing fault     → new peaks at 1.5×, 2×, 3× (harmonic_ratio ↑)
  - Rotor imbalance   → dominant peak shifts
  - Gear wear         → sideband frequencies appear
  - Random noise      → spectral_entropy very high
  - Silence           → spectral_entropy very low, fundamental_power ≈ 0
"""

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_fft_features(
    sequence: np.ndarray,
    sampling_hz: float = 1.0,
    fundamental_hz: float = 10.0 / 60.0,  # 10 cycles / 60 timesteps
) -> np.ndarray:
    """Extract spectral features from a single sequence.

    Args:
        sequence: (T, F) array.
        sampling_hz: Sampling frequency in Hz (1.0 = 1 sample/second).
        fundamental_hz: Expected fundamental frequency of the machinery.

    Returns:
        features: (F * 9,) float32 array.
    """
    n_timesteps, n_sensors = sequence.shape
    n_harmonics = 5
    n_features_per_sensor = 1 + n_harmonics + 3  # fund + 5 harmonics + ratio + dom_freq + entropy = 9
    features = np.zeros(n_sensors * n_features_per_sensor, dtype=np.float32)

    freqs = rfftfreq(n_timesteps, d=1.0 / sampling_hz)

    for i in range(n_sensors):
        s = sequence[:, i]
        base = i * n_features_per_sensor

        fft_coeffs = rfft(s)
        power = np.abs(fft_coeffs) ** 2

        # 1. Fundamental power
        f_idx = int(np.argmin(np.abs(freqs - fundamental_hz)))
        fundamental_power = float(power[f_idx]) if f_idx < len(power) else 0.0
        features[base + 0] = fundamental_power

        # 2-6. Harmonic powers
        harmonic_sum = 0.0
        for n, harmonic_rank in enumerate(range(2, n_harmonics + 2)):
            h_idx = int(np.argmin(np.abs(freqs - fundamental_hz * harmonic_rank)))
            h_power = float(power[h_idx]) if h_idx < len(power) else 0.0
            features[base + 1 + n] = h_power
            harmonic_sum += h_power

        # 7. Harmonic ratio
        features[base + 6] = harmonic_sum / (fundamental_power + 1e-8)

        # 8. Dominant frequency
        features[base + 7] = float(freqs[np.argmax(power)]) if len(power) > 0 else 0.0

        # 9. Spectral entropy
        p_norm = power / (np.sum(power) + 1e-8)
        features[base + 8] = float(-np.sum(p_norm * np.log(p_norm + 1e-8)))

    return features  # (450,)
