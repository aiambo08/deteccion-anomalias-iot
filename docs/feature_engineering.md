# Feature Engineering Documentation

## Overview

Each raw sensor sequence of shape `(60, 50)` is transformed into a 1050-dimensional feature vector combining three complementary feature families:

| Family | Module | Features | Dim |
|---|---|---|---|
| Delta/Derivative | `delta_features.py` | Velocity, acceleration, max delta | 250 |
| Spectral (FFT) | `fft_features.py` | Harmonic power, spectral entropy | 450 |
| Lag/Autocorrelation | `lag_features.py` | Temporal regularity, trend slope | 350 |
| **Total** | `feature_pipeline.py` | | **1050** |

---

## 1. Delta Features (250 dims = 50 sensors × 5)

**Motivation**: Anomalies like spikes and drift produce large temporal derivatives that the raw signal alone may not reveal clearly to a gradient boosting model.

| Feature | Formula | Anomaly Sensitivity |
|---|---|---|
| `delta_1` | `s[t] - s[t-1]` | Spike detection (sudden jump) |
| `delta_3` | `s[t] - s[t-3]` | Short-term trend |
| `delta_5` | `s[t] - s[t-5]` | Medium-term trend |
| `accel`   | `delta_1[t] - delta_1[t-1]` | Drift (sustained acceleration) |
| `max_abs_delta` | `max(|diff(s)|)` over window | Largest anomalous jump |

**How to read**: For a spike anomaly, `max_abs_delta` will be 5–10× larger than in normal sequences. For drift, `accel` will have a consistent positive or negative sign.

---

## 2. FFT Features (450 dims = 50 sensors × 9)

**Motivation**: Mechanical failures often manifest as new frequency components in the signal spectrum before they are visible in the time domain.

| Feature | Description | Anomaly Sensitivity |
|---|---|---|
| `fundamental_power` | Power at expected machinery frequency | Low → sensor failure |
| `harmonic_power_2..5` | Power at 2×, 3×, 4×, 5× fundamental | Bearing/gear wear |
| `harmonic_ratio` | Σ(harmonics) / fundamental | Rotor imbalance indicator |
| `dominant_freq` | Frequency with max spectral power | Frequency shift = fault |
| `spectral_entropy` | Shannon entropy of power spectrum | Very high → noise; very low → silence |

**Physics mapping**:
- Unbalanced rotor → elevated `harmonic_power_2`
- Damaged bearing → elevated `harmonic_power_3` or `harmonic_power_5`
- Gear wear → sideband frequencies
- Signal silence → `spectral_entropy` drops dramatically

---

## 3. Lag/Autocorrelation Features (350 dims = 50 sensors × 7)

**Motivation**: Healthy machinery produces highly predictable (high autocorrelation) signals. Anomalies disrupt this regularity.

| Feature | Description | Normal Range | Anomaly Indicator |
|---|---|---|---|
| `autocorr_lag1` | Pearson correlation at lag=1 | 0.85–0.95 | < 0.40 → chaotic |
| `autocorr_lag3` | Correlation at lag=3 | 0.70–0.90 | Drops significantly |
| `autocorr_lag7` | Correlation at lag=7 | 0.50–0.80 | Near zero → disorders |
| `roll_mean_5`  | Mean of last 5 timesteps | Varies | Shift vs roll_mean_20 |
| `roll_mean_20` | Mean of last 20 timesteps | Varies | Sustained mean change |
| `roll_std_10`  | Std of last 10 timesteps | Low | Elevated → high volatility |
| `linear_trend` | Slope of OLS fit on last 10 steps | ≈ 0 | ≠ 0 → drift |

---

## Validation Procedure

To confirm feature engineering is effective:

```python
from scipy.stats import mannwhitneyu
import numpy as np

X_normal_feats  = np.load("data/features/X_val_advanced.npy")
y_val           = np.load("data/processed/y_val.npy")

n_significant = 0
for feat_idx in range(1050):
    f_norm = X_normal_feats[y_val==0, feat_idx]
    f_anom = X_normal_feats[y_val==1, feat_idx]
    _, p   = mannwhitneyu(f_norm, f_anom, alternative="two-sided")
    if p < 0.05:
        n_significant += 1

pct = n_significant / 1050 * 100
print(f"Significant features: {n_significant}/1050 ({pct:.1f}%)")
# Target: > 50% of features significantly different (p < 0.05)
```

---

## Feature Pipeline Usage

```python
from src.features.feature_pipeline import extract_advanced_features, build_feature_matrix
import numpy as np

# Single sequence
seq = np.random.randn(60, 50)
features = extract_advanced_features(seq)  # (1050,)

# Full dataset
X = np.load("data/processed/X_train.npy")  # (N, 60, 50)
X_features = build_feature_matrix(X, save_path="data/features/X_train_advanced.npy")
```
