# IoT Anomaly Detection — Antigravity

End-to-end ML system for detecting anomalies in industrial IoT sensor data using a **Bi-LSTM Autoencoder** (unsupervised) + **XGBoost** classifier (supervised), with Kafka streaming, FastAPI serving, and MLOps drift monitoring.

---

## Architecture

```
IoT Sensors
    │
    ▼
[Kafka Producer]  ──►  Topic: iot_sensor_data
                               │
                        [Kafka Consumer]
                               │  rolling 60-step buffer per sensor
                               ▼
                    [Bi-LSTM Autoencoder]
                    reconstruction error > threshold?
                               │
              No ◄─────────────┼─────────────► Yes
              │                                  │
          Normal                     Topic: anomaly_alerts
                                          (severity: MEDIUM/HIGH)

Offline:  [XGBoost on 1050 features]  → strong supervised baseline
Hybrid:   60% XGBoost + 40% AE score → best overall performance

MLOps:   Weekly PSI + KS drift checks → automatic retrain trigger
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .          # editable install for src.* imports
```

### 2. Generate & preprocess data

```bash
python -m src.data.generator        # → data/raw/
python -m src.data.preprocessor     # → data/processed/ + models/scaler.pkl
```

### 3. Extract features

```bash
python -m src.features.feature_pipeline   # → data/features/
```

### 4. Train models

```bash
# Bi-LSTM Autoencoder (unsupervised)
python -m src.models.trainer       # → models/bilstm_autoencoder.h5 + models/threshold.pkl

# XGBoost baseline (supervised, uses full train set)
python -m src.models.xgboost_model # → models/xgboost_baseline.json
```

### 5. Run tests

```bash
pytest tests/ -v
```

### 6. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs  (Swagger UI)
```

### 7. Predict (example)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_id": 5, "sequence": [[0.0]*50]*60}'
```

### 8. Full system with Docker Compose

```bash
cd docker
docker-compose up --build
# Services: Zookeeper, Kafka, API, Consumer, Producer
```

---

## Performance Targets

| Metric | Target | Description |
|---|---|---|
| ROC-AUC | ≥ 0.90 | Primary metric |
| PR-AUC | ≥ 0.80 | Critical (5% imbalance) |
| F1-Score | ≥ 0.85 | Balanced accuracy |
| Inference latency | < 50ms | Online detection |
| XGBoost baseline | ≥ 0.95 ROC-AUC | Supervised reference |

---

## Dataset

- **50 sensors**: vibration, temperature, pressure, RPM
- **60 timesteps** per sequence @ 1Hz
- **40,000 sequences** (38,000 normal, 2,000 anomalous — 5%)
- **5 anomaly types**: spike, drift, shift, periodic frequency, silence

---

## Feature Engineering (1050 features)

| Module | Features | Captures |
|---|---|---|
| Delta | 250 | Velocity, acceleration, spike magnitude |
| FFT | 450 | Harmonic spectrum, bearing faults, entropy |
| Lag | 350 | Regularity, drift trend, volatility |

---

## Project Structure

```
├── src/
│   ├── data/          # Generator, preprocessor, dataset utils
│   ├── features/      # Delta, FFT, lag, pipeline
│   ├── models/        # Autoencoder, trainer, evaluator, XGBoost
│   ├── streaming/     # Kafka producer + consumer
│   ├── api/           # FastAPI + schemas + inference engine
│   └── monitoring/    # PSI + KS drift detection + alerts
├── data/              # raw/, processed/, features/, baseline/
├── models/            # Trained model artefacts
├── docker/            # Dockerfiles + docker-compose.yml
├── tests/             # pytest test suite (5 modules)
├── docs/              # Architecture, model card, MLOps runbook
└── notebooks/         # Exploratory notebooks (01-03)
```

---

## MLOps

Weekly drift monitoring:
```bash
python -m src.monitoring.drift_pipeline
```

**Retraining triggered when:**
- ≥ 5 sensors with PSI ≥ 0.25, OR
- ≥ 10 sensors with KS p-value < 0.05

Retrain flag: `logs/drift_reports/retrain_flag_{week}.json`

---

## Anomaly Diagnostics

| Symptom | Likely cause | Action |
|---|---|---|
| Normal error > Anomaly error | Train set contaminated | Filter y_train==0, retrain |
| ROC-AUC < 0.5 | Model inverted | Verify reconstruction direction |
| ROC-AUC 0.5–0.75 | Weak features | Add FFT+Delta features |
| PR-AUC < 0.5 | Bad threshold | Adjust percentile (try p80→p99) |
| PSI > 0.25 in prod | Data drift | Trigger retraining |
