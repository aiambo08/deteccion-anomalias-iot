"""
Train Models — One-shot training script
========================================
Generates all production model artefacts from scratch:

  models/bilstm_autoencoder.h5
  models/scaler.pkl
  models/threshold.pkl
  models/xgboost_baseline.json

Run (from project root, with the venv activated):
    python scripts/train_models.py

Options:
    --ae-only       Train only the Bi-LSTM Autoencoder (skip XGBoost).
    --xgb-only      Train only XGBoost (requires existing scaler.pkl).
    --n-jobs N      Parallel workers for feature extraction [default: 1].
    --skip-gen      Skip data generation if data/raw/X.npy already exists.

Example — fast test run (tiny dataset):
    python scripts/train_models.py --skip-gen --n-jobs 4

Example — full production run:
    python scripts/train_models.py --n-jobs -1
"""

import argparse
import sys
from pathlib import Path

# ── Force UTF-8 output on Windows (avoids UnicodeEncodeError with cp1252) ────
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure the project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train IoT Anomaly Detection models.")
    p.add_argument("--ae-only",  action="store_true", help="Train Autoencoder only.")
    p.add_argument("--xgb-only", action="store_true", help="Train XGBoost only.")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip data generation if data/raw/X.npy already exists.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel jobs for feature extraction (-1 = all cores).")
    return p.parse_args()


# ── Step helpers ──────────────────────────────────────────────────────────────

def generate_data(skip: bool) -> tuple:
    """Return (X, y) either from disk or freshly generated."""
    import numpy as np
    from src.data.generator import IoTDataGenerator  # type: ignore[import]

    raw_x = PROJECT_ROOT / "data" / "raw" / "X.npy"
    raw_y = PROJECT_ROOT / "data" / "raw" / "y.npy"

    if skip and raw_x.exists() and raw_y.exists():
        print(f"\n[Step 1] Loading existing raw data from {raw_x.parent} …")
        X = np.load(raw_x)
        y = np.load(raw_y)
        print(f"  X: {X.shape}  y: {y.shape}  anomaly_rate={y.mean():.2%}")
        return X, y

    print("\n[Step 1] Generating synthetic dataset …")
    gen = IoTDataGenerator(
        n_sensors=50,
        seq_length=60,
        n_sequences=40_000,
        anomaly_ratio=0.05,
        seed=42,
    )
    X, y = gen.generate_dataset()

    raw_x.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np  # noqa: F811 — re-import for clarity
    np.save(raw_x, X)
    np.save(raw_y, y)
    print(f"  Saved raw data → {raw_x.parent}  X:{X.shape}")
    return X, y


def preprocess(X, y) -> dict:
    """Run the full preprocessing pipeline and return splits + scaler path."""
    from src.data.preprocessor import IoTPreprocessor  # type: ignore[import]

    print("\n[Step 2] Preprocessing …")
    prep = IoTPreprocessor(seq_length=60, n_sensors=50)
    splits = prep.full_pipeline(
        X, y,
        save_dir=str(PROJECT_ROOT / "data" / "processed"),
        scaler_path=str(PROJECT_ROOT / "models" / "scaler.pkl"),
    )
    from src.data.dataset import save_baseline_stats  # type: ignore[import]
    save_baseline_stats(
        splits["X_train"],
        path=str(PROJECT_ROOT / "data" / "baseline" / "X_train_baseline_stats.pkl"),
    )
    return splits


def train_autoencoder(splits: dict) -> None:
    """Train and save the Bi-LSTM Autoencoder + threshold."""
    import numpy as np
    from src.models.autoencoder import build_bilstm_autoencoder  # type: ignore[import]
    from src.models.trainer import train_autoencoder as _train, compute_threshold  # type: ignore[import]

    print("\n[Step 3] Training Bi-LSTM Autoencoder …")
    model = build_bilstm_autoencoder(seq_length=60, n_features=50)
    X_val_normal = splits["X_val"][splits["y_val"] == 0]

    model_path = str(PROJECT_ROOT / "models" / "bilstm_autoencoder.h5")
    _train(model, splits["X_train"], X_val_normal, model_path=model_path)

    threshold_path = str(PROJECT_ROOT / "models" / "threshold.pkl")
    thr = compute_threshold(model, X_val_normal, percentile=95, save_path=threshold_path)
    print(f"  Autoencoder threshold (p95): {thr:.6f}")


def train_xgboost(splits: dict, n_jobs: int = 1) -> None:
    """Build feature matrices and train XGBoost classifier."""
    import numpy as np
    from src.data.preprocessor import IoTPreprocessor  # type: ignore[import]
    from src.features.feature_pipeline import build_feature_matrix  # type: ignore[import]
    from src.models.xgboost_model import train_xgboost as _train  # type: ignore[import]

    print("\n[Step 4] Building feature matrices for XGBoost …")

    scaler_path = str(PROJECT_ROOT / "models" / "scaler.pkl")
    prep = IoTPreprocessor()
    prep.load_scaler(scaler_path)

    # Load raw training data (includes anomalies — XGBoost is supervised)
    raw_x = PROJECT_ROOT / "data" / "raw" / "X.npy"
    raw_y = PROJECT_ROOT / "data" / "raw" / "y.npy"
    X_raw = np.load(raw_x)
    y_raw = np.load(raw_y)
    n     = len(X_raw)
    i_tr  = int(n * 0.70)
    X_tr_raw, y_tr = X_raw[:i_tr], y_raw[:i_tr]

    X_tr_scaled  = prep.transform(X_tr_raw)
    X_val_scaled = prep.transform(splits["X_val"])

    parallel_msg = f"n_jobs={n_jobs}" if n_jobs != 1 else "sequential"
    print(f"  Extracting features ({parallel_msg}) …")
    X_train_adv = build_feature_matrix(
        X_tr_scaled, verbose=True, n_jobs=n_jobs,
        save_path=str(PROJECT_ROOT / "data" / "processed" / "X_train_adv.npy"),
    )
    X_val_adv = build_feature_matrix(
        X_val_scaled, verbose=True, n_jobs=n_jobs,
        save_path=str(PROJECT_ROOT / "data" / "processed" / "X_val_adv.npy"),
    )

    print("\n[Step 5] Training XGBoost …")
    xgb_path = str(PROJECT_ROOT / "models" / "xgboost_baseline.json")
    model = _train(
        X_train_adv, y_tr,
        X_val_adv,   splits["y_val"],
        save_path=xgb_path,
    )
    print(f"  XGBoost saved → {xgb_path}")

    from sklearn.metrics import roc_auc_score  # type: ignore[import]
    val_proba = model.predict_proba(X_val_adv)[:, 1]
    roc_auc   = float(roc_auc_score(splits["y_val"], val_proba))
    print(f"  Validation ROC-AUC: {roc_auc:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Ensure models directory exists
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

    run_ae  = not args.xgb_only
    run_xgb = not args.ae_only

    # Data generation and preprocessing are always required unless xgb-only
    if run_ae or (run_xgb and not (PROJECT_ROOT / "data" / "processed" / "X_train.npy").exists()):
        X, y   = generate_data(skip=args.skip_gen)
        splits = preprocess(X, y)
    else:
        # xgb-only with existing processed data
        import numpy as np
        from src.data.dataset import load_splits  # type: ignore[import]
        splits = load_splits(str(PROJECT_ROOT / "data" / "processed"))

    if run_ae:
        train_autoencoder(splits)

    if run_xgb:
        train_xgboost(splits, n_jobs=args.n_jobs)

    print("\n✅  All requested model artefacts have been generated.")
    print(f"   Output directory: {PROJECT_ROOT / 'models'}")


if __name__ == "__main__":
    main()
