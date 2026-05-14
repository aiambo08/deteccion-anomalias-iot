# System Architecture

## Overview

The IoT Anomaly Detection system is a hybrid, production-ready ML pipeline combining unsupervised deep learning (Bi-LSTM Autoencoder) with supervised gradient boosting (XGBoost), deployed through a Kafka streaming pipeline and FastAPI REST service.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                        │
│                                                                           │
│  IoTDataGenerator          IoTPreprocessor          Feature Pipeline     │
│  - 50 sensors              - StandardScaler         - delta_features     │
│  - 5 anomaly types         - temporal split         - fft_features       │
│  - (N, 60, 50) shape       - normal-only train      - lag_features       │
│                                                     → (N, 1050)          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       MODEL LAYER                                         │
│                                                                           │
│  BiLSTM Autoencoder (Keras)          XGBoost Classifier                  │
│  - Encoder: BiLSTM(128) → BiLSTM(64) - scale_pos_weight for 5% ratio    │
│  - Bottleneck: Dense(32)             - early stopping on val AUC         │
│  - Decoder: LSTM(64) → LSTM(128)     - feature importance analysis       │
│  - Threshold: p95 normal val MSE     - ROC-AUC target: > 0.95            │
│                                                                           │
│            Hybrid Ensemble: 60% XGBoost + 40% AE score                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    INFERENCE LAYER                                         │
│                                                                           │
│  Online (Kafka Pipeline):            Offline (FastAPI):                  │
│  - SensorProducer → iot_sensor_data  - POST /predict                     │
│  - AnomalyDetectorConsumer           - rolling buffer maintained in      │
│    • rolling 60-step buffer          - memory per request                │
│    • < 50ms inference target         - GET /alerts/recent                │
│    • → anomaly_alerts topic          - GET /metrics                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    MLOPS LAYER                                            │
│                                                                           │
│  PSIMonitor                          KSMonitor                           │
│  - Population Stability Index        - Kolmogorov-Smirnov test           │
│  - Threshold: PSI ≥ 0.25 → retrain  - Alpha: 0.05 significance          │
│                                                                           │
│  DriftPipeline (weekly orchestration)                                    │
│  - Retrain if: ≥5 PSI critical OR ≥10 KS drifted                       │
│  - Writes retrain_flag_{week}.json                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Raw IoT signals
    │
    ▼ IoTDataGenerator.generate_dataset()
(40000, 60, 50) raw arrays  [data/raw/]
    │
    ▼ IoTPreprocessor.full_pipeline()
(27000, 60, 50) X_train (normal only)   [data/processed/]
(6000,  60, 50) X_val
(7000,  60, 50) X_test
    │
    ├──► Autoencoder training (X_train, X_val_normal only)
    │         │
    │         ▼
    │    models/bilstm_autoencoder.h5
    │    models/threshold.pkl (p95 val MSE)
    │
    └──► Feature extraction → (N, 1050) → XGBoost training
              │
              ▼
         models/xgboost_baseline.json
```

---

## Anomaly Type Detection Capabilities

| Anomaly Type | Primary Detector | Key Feature Signal |
|---|---|---|
| Spike | Autoencoder + Delta | max_abs_delta, delta_1 |
| Drift | XGBoost + Lag | linear_trend, accel |
| Shift | Both | rolling_mean delta, autocorr drop |
| Periodic fault | Autoencoder + FFT | harmonic_ratio, spectral_entropy |
| Silence | Both | rolling_std ≈ 0, spectral_entropy ↓ |

---

## Deployment Topology (Docker Compose)

```
localhost:2181  ── Zookeeper
localhost:9092  ── Kafka Broker
localhost:8000  ── FastAPI API

Docker network (internal): kafka:29092
```
