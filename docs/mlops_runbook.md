# MLOps Runbook — IoT Anomaly Detection System

## 1. System Health Checks

### Check API health
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"BiLSTM_Autoencoder","threshold":...}
```

### Check Kafka topics
```bash
docker exec iot_kafka kafka-topics --bootstrap-server localhost:9092 --list
# Expected: anomaly_alerts, iot_sensor_data
```

### Check recent alerts
```bash
curl http://localhost:8000/alerts/recent?limit=10
```

---

## 2. Training Pipeline

### Full retrain from scratch
```bash
# 1. Regenerate data (if needed)
python -m src.data.generator

# 2. Preprocess
python -m src.data.preprocessor

# 3. Extract features
python -m src.features.feature_pipeline

# 4. Train autoencoder
python -m src.models.trainer

# 5. Train XGBoost baseline
python -m src.models.xgboost_model
```

### Expected artefacts after training
```
models/
├── bilstm_autoencoder.h5
├── xgboost_baseline.json
├── scaler.pkl
└── threshold.pkl
```

---

## 3. Drift Monitoring

### Manual weekly check
```python
import numpy as np
from src.monitoring.drift_pipeline import DriftPipeline
from src.data.dataset import load_splits

splits = load_splits("data/processed")
X_baseline = splits["X_train"]

# Load production data
X_prod = np.load("data/production/X_week_10.npy")

pipeline = DriftPipeline(X_baseline)
report = pipeline.weekly_check(X_prod, week_label="2026-W10")
```

### PSI Thresholds
| PSI | Status | Action |
|---|---|---|
| < 0.10 | OK | Continue monitoring |
| 0.10–0.25 | WARNING | Increase check frequency |
| ≥ 0.25 | CRITICAL | Trigger retraining |

### Retrain decision logic
```
Retrain triggered when ANY of:
  • ≥ 5 sensors with PSI ≥ 0.25
  • ≥ 10 sensors with KS p-value < 0.05
```

---

## 4. Model Update Procedure

1. Retrain flag found at `logs/drift_reports/retrain_flag_{week}.json`
2. Collect new labelled data (or expand existing dataset)
3. Re-run full training pipeline (section 2)
4. Evaluate new model: `python -m src.models.evaluator` — confirm ROC-AUC ≥ 0.90
5. Replace `models/bilstm_autoencoder.h5` and `models/threshold.pkl`
6. Restart API service: `docker-compose restart api consumer`
7. Monitor for 48h for regression

---

## 5. Diagnostic Decision Tree

```
Performance degraded?
    │
    ├── Normal error > Anomaly error?
    │       YES → Training data contaminated with anomalies
    │             FIX: Filter y_train==0, retrain
    │
    ├── ROC-AUC < 0.50?
    │       YES → Model critically broken
    │             FIX: Verify data pipeline end-to-end
    │
    ├── ROC-AUC 0.50–0.75?
    │       YES → Weak discrimination
    │             FIX: Add/tune features; increase model capacity
    │
    ├── PR-AUC < 0.50?
    │       YES → Threshold miscalibrated
    │             FIX: Adjust percentile (80 → 90 → 95 → 99)
    │
    └── PSI > 0.25 in production?
            YES → Data drift
                  FIX: Retrain with recent data
```

---

## 6. Log Locations

| Log type | Location |
|---|---|
| Training losses | `logs/training/` |
| Drift reports | `logs/drift_reports/drift_{week}_{time}.json` |
| Retrain flags | `logs/drift_reports/retrain_flag_{week}.json` |
| Anomaly alerts | `logs/anomaly_alerts/alerts_{date}.log` |
| Evaluation plots | `logs/evaluation_report.png` |

---

## 7. Docker Operations

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Stop all services
docker-compose -f docker/docker-compose.yml down

# View consumer logs
docker logs -f iot_consumer

# View API logs
docker logs -f iot_api

# Restart single service after model update
docker-compose -f docker/docker-compose.yml restart api
```
