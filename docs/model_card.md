# Model Card — Bi-LSTM Autoencoder for IoT Anomaly Detection

## Model Details

| Attribute | Value |
|---|---|
| Name | BiLSTM_Autoencoder |
| Version | 1.0.0 |
| Type | Sequence-to-Sequence Autoencoder (Unsupervised) |
| Framework | TensorFlow / Keras |
| Input | (batch, 60, 50) — 60 timesteps × 50 sensors |
| Output | (batch, 60, 50) — Reconstructed sequence |
| Parameters | ~3.5M trainable |
| Loss | Mean Squared Error (MSE) |

## Architecture

```
Input (60, 50)
→ BiLSTM(128, return_sequences=True)  + Dropout(0.2)
→ BiLSTM(64,  return_sequences=False) + Dropout(0.2)
→ Dense(32, relu)  ← BOTTLENECK
→ RepeatVector(60)
→ LSTM(64,  return_sequences=True)  + Dropout(0.2)
→ LSTM(128, return_sequences=True)
→ TimeDistributed(Dense(50))
Output (60, 50)
```

## Intended Use

- **Primary use**: Detecting anomalous behaviour in industrial IoT sensor streams
- **Detection mechanism**: High reconstruction error → anomaly
- **Anomaly types**: Spike, drift, regime shift, periodic fault, sensor silence

## Training Data

- **Domain**: Synthetic industrial IoT signals (vibration, temperature, pressure, RPM)
- **Training set**: Normal sequences only (27,000 × ~70% of 38,000 normals)
- **Train contamination**: ZERO anomalies (enforced by y_train filter)
- **Normalisation**: StandardScaler fitted on train normals only

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| ROC-AUC | ≥ 0.90 | Primary metric |
| PR-AUC | ≥ 0.80 | Critical with 5% anomaly rate |
| F1-Score | ≥ 0.85 | At optimal threshold |
| Inference latency | < 50ms | Per sequence, GPU optional |

## Threshold Computation

```
threshold = np.percentile(val_normal_reconstruction_errors, 95)
```

The p95 threshold is deliberately conservative — only the top 5% of normal
reconstruction errors would exceed it, limiting false positives.

## Limitations

1. **Unseen anomaly types**: May not detect novel anomaly patterns not present in training
2. **Concept drift**: Must be retriggered if normal operating regime changes significantly
3. **Single threshold**: A global threshold may need sensor-specific tuning in real deployments
4. **Latency vs accuracy**: Bidirectional LSTM cannot be used for true online (causal) inference. The 60-step buffer introduces ~1 minute lag at 1Hz.
5. **Synthetic data**: Performance on real hardware may differ from synthetic benchmarks

## Ethical Considerations

- **Safety**: False negatives (missed anomalies) in industrial settings can be dangerous. Threshold should be tuned to prioritise recall over precision.
- **Accountability**: Human review is required before automated maintenance actions are triggered.
- **Data privacy**: Sensor data may be commercially sensitive; encrypt at rest and in transit.

## Monitoring & Maintenance

- **Drift detection**: Weekly PSI + KS tests on all 50 sensors
- **Retrain trigger**: PSI ≥ 0.25 on ≥5 sensors, or KS p < 0.05 on ≥10 sensors
- **Model evaluation**: Re-evaluate ROC-AUC and PR-AUC after each retrain

## How to Use

```python
from src.models.trainer import load_model_and_threshold
model, threshold = load_model_and_threshold()

# Predict on one sequence
import numpy as np
X = np.load("data/processed/X_test.npy")[:1]  # (1, 60, 50)
recon = model.predict(X, verbose=0)
error = np.mean((X - recon) ** 2)
is_anomaly = error > threshold
print(f"Anomaly: {is_anomaly}  Error: {error:.6f}")
```
