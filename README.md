# IoT Anomaly Detection

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
                    [Hybrid Ensemble Inference]
                    60% XGBoost + 40% AE score
                    hybrid_score > 0.5?
                               │
              No ◄─────────────┼─────────────► Yes
              │                                  │
          Normal                     Topic: anomaly_alerts
                                          (severity: MEDIUM/HIGH)

Offline:  [XGBoost on 1050 features]  → strong supervised baseline
Hybrid:   60% XGBoost + 40% AE score → best overall performance

MLOps:   Weekly PSI + KS drift checks → automatic retrain trigger
         retrain_executor watches for flag → compares ROC-AUC → promotes
```

---

## Performance Targets

| Metric | Target | Description |
|---|---|---|
| ROC-AUC | ≥ 0.90 | Primary metric |
| PR-AUC | ≥ 0.80 | Critical (5% class imbalance) |
| F1-Score | ≥ 0.85 | Balanced accuracy |
| Inference latency | < 50 ms | Online detection SLA |
| XGBoost baseline | ≥ 0.95 ROC-AUC | Supervised reference |

---

## Execution & Operations Guide (Windows)

> **All commands are for PowerShell on Windows.** Run from the project root directory.

---

### 1. Environment Setup

#### Prerequisites

- **Python 3.12** — TensorFlow ≥ 2.16 requires Python ≤ 3.12 (not 3.13)
- **Docker Desktop** — for production deployment
- **uv** — fast Python package manager

#### 1.1 Install `uv`

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
uv --version
```

#### 1.2 Create the Virtual Environment (Python 3.12)

```powershell
# Install Python 3.12 via uv (if not already available)
uv python install 3.12

# Create the .venv with Python 3.12 (required for TensorFlow)
uv venv --python 3.12 .venv
```

> **Why Python 3.12?** TensorFlow does not publish wheels for Python 3.13 yet.
> Running `uv python install 3.12` takes ~30 s and installs an isolated CPython binary.

#### 1.3 Install Dependencies

```powershell
# Install all runtime dependencies into the .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# Install the project in editable mode (required for src.* imports)
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
```

> The editable install (`-e .`) is **mandatory**. Without it, all
> `from src.*` imports fail with `ModuleNotFoundError`.

#### 1.4 Activate the Environment

```powershell
.venv\Scripts\Activate.ps1
```

From this point, all commands in this guide assume the venv is active.
If you close the terminal, re-run the activation line.

Alternatively, prefix any command with the full Python path to avoid activating:

```powershell
& ".venv\Scripts\python.exe" <script>
```

#### 1.5 Key Dependencies

| Package | Version | Role |
|---|---|---|
| `tensorflow` | ≥ 2.16.0 | Bi-LSTM Autoencoder |
| `xgboost` | ≥ 1.7.0 | Supervised baseline + hybrid ensemble |
| `fastapi` + `uvicorn` | latest | REST API serving |
| `kafka-python` | latest | Streaming consumer/producer |
| `scikit-learn` | ≥ 1.3.0 | Scaler, metrics |
| `joblib` | ≥ 1.3.0 | Parallel feature extraction |

---

### 2. Training Pipeline — Generate All Model Artefacts

Use the unified training script `scripts/train_models.py`.
It covers data generation, preprocessing, autoencoder training, and XGBoost training in one command.

> **Note on Unicode in PowerShell:** Some print statements use Unicode arrows.
> Prefix commands with `$env:PYTHONUTF8=1;` to avoid encoding errors on Windows.

#### Option A — Full pipeline from scratch (first run)

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" scripts/train_models.py
```

This runs all five steps sequentially:

| Step | Action | Output |
|---|---|---|
| 1 | Generate 40,000 synthetic sequences | `data/raw/X.npy`, `data/raw/y.npy` |
| 2 | Preprocess + fit scaler | `data/processed/X_train.npy` … `scaler.pkl` |
| 3 | Train Bi-LSTM Autoencoder | `models/bilstm_autoencoder.h5`, `models/threshold.pkl` |
| 4 | Extract 1050-feature matrices | `data/processed/X_train_adv.npy` … |
| 5 | Train XGBoost | `models/xgboost_baseline.json` |

#### Option B — Skip data generation (raw data already exists)

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" scripts/train_models.py --skip-gen
```

#### Option C — XGBoost only (autoencoder artefacts already exist)

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" scripts/train_models.py --xgb-only --skip-gen --n-jobs -1
```

`--n-jobs -1` uses all available CPU cores for feature extraction.

#### Option D — Autoencoder only

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" scripts/train_models.py --ae-only --skip-gen
```

#### Required artefacts after training

All four files must be present before starting the API or consumer:

```
models/
├── bilstm_autoencoder.h5    ← Keras Bi-LSTM model
├── xgboost_baseline.json    ← XGBoost classifier
├── scaler.pkl               ← Fitted StandardScaler
└── threshold.pkl            ← p95 MSE anomaly threshold
```

Verify:

```powershell
Get-ChildItem models\
# Must list: bilstm_autoencoder.h5  xgboost_baseline.json  scaler.pkl  threshold.pkl
```

---

### 3. Running the Pipeline Step by Step (Manual)

If you prefer individual control over each step:

#### Step 1 — Data Generation

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.data.generator
```

Output: `data/raw/X.npy` (40000, 60, 50), `data/raw/y.npy` (40000,)

#### Step 2 — Preprocessing

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.data.preprocessor
```

Output: `data/processed/` splits + `models/scaler.pkl`

#### Step 3 — Feature Engineering

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.features.feature_pipeline
```

Output: `data/features/X_train_advanced.npy` (N, 1050)

> Feature extraction is CPU-intensive (~10–25 min sequential).
> Use `build_feature_matrix(n_jobs=-1)` in code for parallel extraction.

#### Step 4a — Bi-LSTM Autoencoder

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.models.trainer
```

Output: `models/bilstm_autoencoder.h5`, `models/threshold.pkl`

#### Step 4b — XGBoost Classifier

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.models.xgboost_model
```

Output: `models/xgboost_baseline.json`

#### Step 5 — Evaluate

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.models.evaluator
```

Output: `logs/evaluation_report.png` and console metrics.

Expected results:
```
Normal sequences    — mean MSE: ~0.002
Anomalous sequences — mean MSE: ~0.018
Threshold (p95)     : 0.0045
ROC-AUC             : >= 0.90
```

> **Red flag:** If `Normal MSE > Anomaly MSE`, training data contains anomalous sequences.
> Re-run the preprocessor and verify `y_train` contains only zeros.

---

### 4. Testing

```powershell
# Full test suite
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m pytest tests/ -v

# Specific module
& ".venv\Scripts\python.exe" -m pytest tests/test_api.py -v

# With coverage
& ".venv\Scripts\python.exe" -m pytest tests/ -v --cov=src --cov-report=term-missing
```

| Test module | Covers |
|---|---|
| `test_preprocessing.py` | Generator shape, anomaly ratio, split ratios |
| `test_features.py` | Feature dimensions, delta/FFT/lag correctness |
| `test_model.py` | Autoencoder forward pass, threshold computation |
| `test_api.py` | FastAPI endpoints, CORS, X-Admin-Key auth |
| `test_monitoring.py` | PSI bounds, KS p-value correctness |
| `test_kafka.py` | Producer/consumer message format |

---

### 5. Local Development — Running Without Docker

For development without Docker, start each component in a separate PowerShell window.

> Requires a running Kafka broker. Use the Docker Kafka-only stack:
> ```powershell
> docker-compose -f docker/docker-compose.yml up -d zookeeper kafka
> ```

**Terminal 1 — FastAPI server:**

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Kafka consumer:**

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.streaming.consumer
```

**Terminal 3 — Kafka producer (simulates IoT sensors):**

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.streaming.producer
```

**Terminal 4 — Retrain executor (watch mode):**

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.monitoring.retrain_executor --watch
```

---

### 6. Production Deployment — Docker Orchestration

#### 6.1 Configure Environment Variables

Before starting, set the admin API key for the `DELETE /alerts` endpoint:

```powershell
$env:ADMIN_API_KEY = "your-strong-secret-key"
```

Or create a `.env` file at the project root:

```env
ADMIN_API_KEY=your-strong-secret-key
ALLOWED_ORIGINS=https://your-dashboard-domain.com
MAX_RETRAIN_HOURS=6
```

#### 6.2 Pre-flight Check

```powershell
# Verify all model artefacts exist
Get-ChildItem models\
# Must show: bilstm_autoencoder.h5  xgboost_baseline.json  scaler.pkl  threshold.pkl
```

#### 6.3 Start the Full Stack

```powershell
docker-compose -f docker/docker-compose.yml up --build -d
```

**Service startup order** (enforced by healthcheck dependencies):

```
1. Zookeeper      (port 2181)  ← starts first
        │ service_healthy
2. Kafka          (port 9092)  ← waits for Zookeeper
        │ service_healthy (30 s start_period + broker-api probe)
3. FastAPI API    (port 8000)  ← loads model artefacts
        │ service_healthy
4. Consumer                    ← waits for Kafka + API
5. Producer                    ← waits for Kafka
6. retrain_executor            ← watch mode, waits for API
```

#### 6.4 Verify Container Health

```powershell
docker-compose -f docker/docker-compose.yml ps
```

Expected (all healthy or running):

```
NAME                    STATUS
iot_zookeeper           Up   (healthy)
iot_kafka               Up   (healthy)
iot_api                 Up   (healthy)
iot_consumer            Up
iot_producer            Up
iot_retrain_executor    Up
```

#### 6.5 Follow Logs

```powershell
# API logs (inference, anomaly detections)
docker logs -f iot_api

# Consumer logs (per-sensor rolling buffer)
docker logs -f iot_consumer

# Retraining executor
docker logs -f iot_retrain_executor

# Kafka topic list
docker exec iot_kafka kafka-topics --bootstrap-server localhost:9092 --list
# Expected: anomaly_alerts  iot_sensor_data
```

#### 6.6 Stop / Restart

```powershell
# Stop all services (preserves volumes)
docker-compose -f docker/docker-compose.yml down

# Full clean slate (removes volumes)
docker-compose -f docker/docker-compose.yml down -v

# Restart a single service after model update
docker-compose -f docker/docker-compose.yml restart api consumer
```

---

### 7. API Interaction

Swagger UI: **http://localhost:8000/docs**

#### 7.1 Health Check

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

```json
{
  "status": "ok",
  "model": "Hybrid_BiLSTM_XGBoost",
  "threshold": 0.004521,
  "mode": "hybrid",
  "version": "2.0.0"
}
```

`mode` is `"autoencoder"` if `xgboost_baseline.json` is missing — the API degrades gracefully.

#### 7.2 Inference — POST /predict

```powershell
# Build a (60, 50) zero-sequence payload
$seq = New-Object System.Collections.Generic.List[object]
1..60 | ForEach-Object { $seq.Add( (New-Object double[] 50) ) }
$body = @{ sensor_id = 5; sequence = $seq } | ConvertTo-Json -Depth 4

Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json
```

```json
{
  "sensor_id": 5,
  "is_anomaly": false,
  "reconstruction_error": 0.00198432,
  "threshold": 0.00452100,
  "hybrid_score": 0.043210,
  "xgb_proba": 0.031400,
  "severity": "NONE",
  "inference_ms": 18.4,
  "mode": "hybrid"
}
```

| Field | Description |
|---|---|
| `is_anomaly` | `true` if `hybrid_score > 0.5` |
| `severity` | `NONE` / `MEDIUM` / `HIGH` (≥ 2× threshold) |
| `hybrid_score` | `0.60 × xgb_proba + 0.40 × ae_norm` |
| `mode` | `hybrid` when XGBoost loaded, else `autoencoder` |

#### 7.3 Recent Alerts

```powershell
Invoke-RestMethod "http://localhost:8000/alerts/recent?limit=20" | ConvertTo-Json
```

#### 7.4 Aggregate Metrics

```powershell
Invoke-RestMethod http://localhost:8000/metrics | ConvertTo-Json
```

#### 7.5 Clear Alerts (Admin)

```powershell
Invoke-RestMethod -Uri http://localhost:8000/alerts `
  -Method Delete `
  -Headers @{ "X-Admin-Key" = $env:ADMIN_API_KEY }
```

Returns `401` if the key is wrong, `503` if `ADMIN_API_KEY` is not set.

#### 7.6 Drift Status

```powershell
Invoke-RestMethod http://localhost:8000/drift/status | ConvertTo-Json
```

Returns `{ "available": false }` until the first drift report is generated.

---

### 8. MLOps: Drift Monitoring & Retraining

#### 8.1 Run Drift Check Manually

```powershell
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -c "
import numpy as np
from src.monitoring.drift_pipeline import DriftPipeline
from src.data.dataset import load_splits

splits = load_splits('data/processed')
X_prod = np.load('data/production/X_week_10.npy')   # current production data

pipeline = DriftPipeline(splits['X_train'])
report   = pipeline.weekly_check(X_prod, week_label='2026-W10')
"
```

Outputs to `logs/drift_reports/`:

```
drift_report_2026-W10_120000.json     ← PSI + KS metrics
retrain_flag_2026-W10.json            ← written only if thresholds exceeded
```

#### 8.2 PSI Thresholds

| PSI Value | Status | Action |
|---|---|---|
| < 0.10 | Stable | Continue normal monitoring |
| 0.10 – 0.25 | Warning | Increase check frequency |
| ≥ 0.25 | Critical | Trigger retraining |

Retraining auto-triggers when **either** condition is met:
- ≥ 5 sensors with PSI ≥ 0.25
- ≥ 10 sensors with KS p < 0.05

#### 8.3 Retraining Executor

```powershell
# Process all pending retrain flags (single-shot)
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.monitoring.retrain_executor

# Continuous watch mode (production daemon)
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.monitoring.retrain_executor --watch
```

Set `MAX_RETRAIN_HOURS` to get a warning if retraining takes too long:

```powershell
$env:MAX_RETRAIN_HOURS = "4"
```

**Promotion logic:**

```
Candidate trained
    │
    ├── No production model? → Promote unconditionally
    │
    └── Δ ROC-AUC ≥ 0.005?
              YES → Archive old models to models/archive/{week}/
                    Promote: autoencoder + xgboost + scaler + threshold
              NO  → Keep production model, write skip report
```

> The scaler is refitted on the current training data at each retraining cycle,
> keeping normalisation aligned with the live data distribution.

#### 8.4 Force Manual Update

```powershell
# 1. Retrain all models
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" scripts/train_models.py --skip-gen --n-jobs -1

# 2. Validate ROC-AUC >= 0.90
$env:PYTHONUTF8=1; & ".venv\Scripts\python.exe" -m src.models.evaluator

# 3. Restart production services
docker-compose -f docker/docker-compose.yml restart api consumer

# 4. Verify new threshold loaded
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

#### 8.5 Log Locations

| Log type | Path |
|---|---|
| Drift reports | `logs/drift_reports/drift_report_{week}_{ts}.json` |
| Retrain flags | `logs/drift_reports/retrain_flag_{week}.json` |
| Retrain reports | `logs/retrain_reports/retrain_report_{week}_{ts}.json` |
| Anomaly alerts | `logs/alerts.db` (SQLite) |
| Evaluation plots | `logs/evaluation_report.png` |

---

### 9. Troubleshooting

#### 9.1 Common Errors on Windows

| Error | Cause | Fix |
|---|---|---|
| `Distribution tensorflow can't be installed` | Python 3.13 — no TF wheels | Use `uv venv --python 3.12 --clear .venv` and reinstall |
| `UnicodeEncodeError: 'charmap' codec` | Console is cp1252 | Prefix command with `$env:PYTHONUTF8=1;` |
| `ModuleNotFoundError: No module named 'src'` | Editable install missing | `uv pip install --python .venv\Scripts\python.exe -e . --no-deps` |
| `No Python at '...python.exe'` | Wrong venv / Anaconda path conflict | Use `& ".venv\Scripts\python.exe"` explicitly instead of `python` |
| `Failed to build setuptools.backends.legacy` | Old build backend spec | `build-backend = "setuptools.build_meta"` in `pyproject.toml` |
| `NoBrokersAvailable` in consumer | Kafka not ready | Wait for `iot_kafka` healthy: `docker ps` |
| `Model not found` at API startup | Missing artefacts | Run `scripts/train_models.py` |

#### 9.2 Verify Python Version in Active Environment

```powershell
& ".venv\Scripts\python.exe" --version
# Must show: Python 3.12.x
```

#### 9.3 Verify TensorFlow Installed Correctly

```powershell
& ".venv\Scripts\python.exe" -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.16.x or later
```

#### 9.4 Diagnostic Decision Tree

```
Model performance degraded?
    │
    ├─► Normal MSE > Anomaly MSE?
    │       Training data contaminated with anomalies.
    │       Fix: python -c "import numpy as np; y=np.load('data/processed/y_train.npy'); print(y.sum())"
    │            Expected: 0.  If non-zero → re-run preprocessor.
    │
    ├─► ROC-AUC < 0.50?
    │       Model critically broken (inverted predictions).
    │       Fix: Check data pipeline shapes end-to-end.
    │
    ├─► ROC-AUC 0.50–0.75?
    │       Weak feature discrimination.
    │       Fix: python -c "import numpy as np; X=np.load('data/processed/X_val_adv.npy'); print(X.shape)"
    │            Expected: (N, 1050).
    │
    ├─► PSI >= 0.25 in production?
    │       Data drift detected.
    │       Fix: python -m src.monitoring.retrain_executor --watch
    │
    └─► hybrid_score always ~0.5?
            AE normaliser not yet warmed up.
            Normal — EMA stabilises after ~20 sequences.
```

---

## Dataset

- **50 sensors**: vibration, temperature, pressure, RPM
- **60 timesteps** per sequence @ 1 Hz
- **40,000 sequences** (38,000 normal, 2,000 anomalous — 5%)
- **5 anomaly types**: spike, drift, shift, periodic frequency fault, silence

---

## Feature Engineering (1050 features)

| Module | Shape | Captures |
|---|---|---|
| Delta | 250 = 50 × 5 | Velocity, acceleration, spike magnitude |
| FFT | 450 = 50 × 9 | Harmonic spectrum, bearing faults, spectral entropy |
| Lag | 350 = 50 × 7 | Regularity, drift trend, autocorrelation, volatility |

---

## Project Structure

```
├── scripts/
│   └── train_models.py    # Unified training script (all artefacts)
├── src/
│   ├── data/              # Generator, preprocessor, dataset utils
│   ├── features/          # Delta, FFT, lag extractors + pipeline
│   ├── models/            # Autoencoder, trainer, evaluator, XGBoost
│   ├── streaming/         # Kafka producer + consumer (Hybrid Ensemble)
│   ├── api/               # FastAPI + Pydantic schemas + inference engine
│   └── monitoring/        # PSI + KS drift detection + retrain executor
├── data/                  # raw/, processed/, features/, production/
├── models/                # Trained artefacts + archive/
├── docker/                # Dockerfiles + docker-compose.yml
├── tests/                 # pytest suite (6 modules)
├── logs/                  # Drift reports, alerts.db, plots
└── notebooks/             # Exploratory notebooks
```
