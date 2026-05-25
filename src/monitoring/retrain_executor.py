"""
MLOps Retraining Executor
=========================
Watches for ``retrain_flag_{week}.json`` files written by ``DriftPipeline``
and orchestrates the full retraining + model-promotion workflow:

  1. Load the latest data splits and preprocessor.
  2. Re-run the Bi-LSTM Autoencoder training pipeline.
  3. Re-run XGBoost on advanced feature vectors.
  4. Evaluate both new models on the held-out test set.
  5. Compare new ROC-AUC against the current production model.
  6. If improvement ≥ MIN_IMPROVEMENT_DELTA:
       a. Archive old model artefacts to ``models/archive/{week}/``.
       b. Copy new artefacts to ``models/`` (production).
  7. Write a promotion report to ``logs/retrain_reports/``.

Usage (stand-alone):
    python -m src.monitoring.retrain_executor

Usage (continuous watch mode):
    python -m src.monitoring.retrain_executor --watch
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REPORT_DIR          = Path("logs/drift_reports")
RETRAIN_REPORT_DIR  = Path("logs/retrain_reports")
MODELS_DIR          = Path("models")
ARCHIVE_DIR         = MODELS_DIR / "archive"

PROCESSED_DIR       = Path("data/processed")
RAW_DIR             = Path("data/raw")

# Minimum absolute ROC-AUC gain required to promote the new model
MIN_IMPROVEMENT_DELTA = 0.005

# Poll interval for --watch mode (seconds)
WATCH_INTERVAL_SECONDS = 60

# Maximum expected retraining time in hours; a WARNING is emitted if exceeded.
# Set via env var MAX_RETRAIN_HOURS (e.g. "4" for 4 h).
MAX_RETRAIN_HOURS: float = float(os.environ.get("MAX_RETRAIN_HOURS", "6"))

# ──────────────────────────────────────────────────────────────────────────────
# Artefact paths
# ──────────────────────────────────────────────────────────────────────────────

PRODUCTION_ARTEFACTS = {
    "autoencoder":  MODELS_DIR / "bilstm_autoencoder.h5",
    "xgboost":      MODELS_DIR / "xgboost_baseline.json",
    "scaler":       MODELS_DIR / "scaler.pkl",
    "threshold":    MODELS_DIR / "threshold.pkl",
}

CANDIDATE_ARTEFACTS = {
    "autoencoder": MODELS_DIR / "candidate_bilstm_autoencoder.h5",
    "xgboost":     MODELS_DIR / "candidate_xgboost.json",
    "threshold":   MODELS_DIR / "candidate_threshold.pkl",
    "scaler":      MODELS_DIR / "candidate_scaler.pkl",   # I2: updated scaler
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Discover pending retrain flags
# ──────────────────────────────────────────────────────────────────────────────

def find_pending_flags() -> list[Path]:
    """Return all unprocessed ``retrain_flag_*.json`` files, sorted by mtime."""
    pattern = str(REPORT_DIR / "retrain_flag_*.json")
    flags   = sorted(glob.glob(pattern), key=os.path.getmtime)
    done_marker = RETRAIN_REPORT_DIR / ".processed_flags"
    processed   = set()
    if done_marker.exists():
        processed = set(done_marker.read_text().splitlines())
    return [Path(f) for f in flags if Path(f).name not in processed]


def mark_flag_processed(flag_path: Path) -> None:
    """Record that this flag file has been handled to avoid re-runs."""
    RETRAIN_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    done_marker = RETRAIN_REPORT_DIR / ".processed_flags"
    with done_marker.open("a") as fh:
        fh.write(flag_path.name + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Full training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _load_data() -> dict:
    """Load processed splits and raw arrays from disk."""
    from src.data.dataset import load_splits  # type: ignore[import]

    splits = load_splits(str(PROCESSED_DIR))

    # Raw arrays needed for XGBoost (requires labelled data)
    X_raw = np.load(RAW_DIR / "X.npy")
    y_raw = np.load(RAW_DIR / "y.npy")

    n      = len(X_raw)
    i_tr   = int(n * 0.70)
    X_tr_raw, y_tr_all = X_raw[:i_tr], y_raw[:i_tr]

    return {**splits, "X_tr_raw": X_tr_raw, "y_tr_all": y_tr_all}


def _run_autoencoder_pipeline(data: dict) -> float:
    """Retrain Bi-LSTM Autoencoder; return new threshold, save candidate."""
    from src.models.autoencoder import build_bilstm_autoencoder  # type: ignore[import]
    from src.models.trainer import train_autoencoder, compute_threshold  # type: ignore[import]

    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_val_n = X_val[y_val == 0]

    model = build_bilstm_autoencoder()
    logger.info("Training Bi-LSTM Autoencoder on %d normal sequences…", len(data["X_train"]))
    train_autoencoder(
        model,
        data["X_train"],
        X_val_n,
        model_path=str(CANDIDATE_ARTEFACTS["autoencoder"]),
    )
    threshold = compute_threshold(
        model,
        X_val_n,
        save_path=str(CANDIDATE_ARTEFACTS["threshold"]),
    )
    logger.info("Candidate autoencoder threshold: %.6f", threshold)
    return threshold


def _run_xgboost_pipeline(data: dict) -> float:
    """Retrain XGBoost with a refreshed scaler; return validation ROC-AUC.

    The scaler is **refitted on the current training data** (I2 fix) so that
    the normalisation parameters reflect the live data distribution rather than
    the original baseline.  The new scaler is saved as a candidate artefact
    alongside the candidate XGBoost model.
    """
    from src.data.preprocessor import IoTPreprocessor  # type: ignore[import]
    from src.features.feature_pipeline import build_feature_matrix  # type: ignore[import]
    from src.models.xgboost_model import train_xgboost  # type: ignore[import]
    from sklearn.metrics import roc_auc_score  # type: ignore[import]

    prep = IoTPreprocessor()

    # I2 — Refit the scaler on NORMAL training sequences only (same as original pipeline).
    # This ensures normalisation stays aligned with the current data distribution.
    X_tr_raw = data["X_tr_raw"]
    y_tr_all = data["y_tr_all"]
    X_tr_normal = X_tr_raw[y_tr_all == 0]

    logger.info(
        "Refitting StandardScaler on %d normal training sequences …", len(X_tr_normal)
    )
    X_tr_scaled = prep.fit_transform_train(X_tr_normal)
    prep.save_scaler(str(CANDIDATE_ARTEFACTS["scaler"]))
    logger.info("Candidate scaler saved → %s", CANDIDATE_ARTEFACTS["scaler"])

    # Transform validation set with the refreshed scaler
    X_val_scaled = prep.transform(data["X_val"])

    logger.info("Building candidate feature matrices …")
    X_train_adv = build_feature_matrix(X_tr_scaled, verbose=True)
    X_val_adv   = build_feature_matrix(X_val_scaled, verbose=True)

    model = train_xgboost(
        X_train_adv, y_tr_all,
        X_val_adv,   data["y_val"],
        save_path=str(CANDIDATE_ARTEFACTS["xgboost"]),
    )

    val_proba = model.predict_proba(X_val_adv)[:, 1]
    roc_auc   = float(roc_auc_score(data["y_val"], val_proba))
    logger.info("Candidate XGBoost ROC-AUC: %.4f", roc_auc)
    return roc_auc


def run_full_training(data: dict) -> dict:
    """Execute the end-to-end retraining and return a metrics summary.

    Emits a WARNING log if the total elapsed time exceeds MAX_RETRAIN_HOURS
    (M5 — timeout observability).
    """
    logger.info("═" * 65)
    logger.info("  FULL RETRAINING PIPELINE — %s", datetime.now().isoformat())
    logger.info("═" * 65)

    t_start = time.perf_counter()

    threshold   = _run_autoencoder_pipeline(data)
    new_roc_auc = _run_xgboost_pipeline(data)

    elapsed_h = (time.perf_counter() - t_start) / 3600
    logger.info("Total retraining time: %.2f h", elapsed_h)
    if elapsed_h > MAX_RETRAIN_HOURS:
        logger.warning(
            "Retraining exceeded MAX_RETRAIN_HOURS (%.1f h > %.1f h). "
            "Consider reducing dataset size, epochs, or adding a training time limit.",
            elapsed_h,
            MAX_RETRAIN_HOURS,
        )

    return {
        "candidate_threshold": threshold,
        "candidate_roc_auc":   new_roc_auc,
        "elapsed_hours":       round(elapsed_h, 3),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Evaluate current production model
# ──────────────────────────────────────────────────────────────────────────────

def _production_roc_auc(data: dict) -> Optional[float]:
    """Return the ROC-AUC of the current production XGBoost on the val split."""
    xgb_path = PRODUCTION_ARTEFACTS["xgboost"]
    if not xgb_path.exists():
        logger.warning("No production XGBoost found at %s — treating as 0.0", xgb_path)
        return None

    try:
        import xgboost as xgb  # type: ignore[import]
        from src.data.preprocessor import IoTPreprocessor  # type: ignore[import]
        from src.features.feature_pipeline import build_feature_matrix  # type: ignore[import]
        from sklearn.metrics import roc_auc_score  # type: ignore[import]

        prod_model = xgb.XGBClassifier()
        prod_model.load_model(str(xgb_path))

        prep = IoTPreprocessor()
        prep.load_scaler()
        X_val_adv = build_feature_matrix(data["X_val"], verbose=False)

        val_proba = prod_model.predict_proba(X_val_adv)[:, 1]
        roc_auc   = float(roc_auc_score(data["y_val"], val_proba))
        logger.info("Production XGBoost ROC-AUC: %.4f", roc_auc)
        return roc_auc
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not evaluate production model: %s", exc)
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Model promotion
# ──────────────────────────────────────────────────────────────────────────────

def _archive_production(week_label: str) -> None:
    """Move current production artefacts to ``models/archive/{week}/``."""
    dest = ARCHIVE_DIR / week_label
    dest.mkdir(parents=True, exist_ok=True)

    for name, src_path in PRODUCTION_ARTEFACTS.items():
        if src_path.exists():
            shutil.copy2(src_path, dest / src_path.name)
            logger.info("Archived %s → %s", src_path.name, dest)


def _promote_candidate(week_label: str) -> None:
    """Copy candidate artefacts to production paths.

    Includes the candidate scaler (I2) so that the production normalisation
    parameters stay aligned with the newly trained model.
    """
    _archive_production(week_label)

    mapping = {
        CANDIDATE_ARTEFACTS["autoencoder"]: PRODUCTION_ARTEFACTS["autoencoder"],
        CANDIDATE_ARTEFACTS["xgboost"]:     PRODUCTION_ARTEFACTS["xgboost"],
        CANDIDATE_ARTEFACTS["threshold"]:   PRODUCTION_ARTEFACTS["threshold"],
        CANDIDATE_ARTEFACTS["scaler"]:      PRODUCTION_ARTEFACTS["scaler"],  # I2
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Promoted %s → %s", src.name, dst)
        else:
            logger.warning("Candidate artefact missing: %s", src)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Report persistence
# ──────────────────────────────────────────────────────────────────────────────

def _write_report(report: dict, week_label: str) -> Path:
    RETRAIN_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = RETRAIN_REPORT_DIR / f"retrain_report_{week_label}_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report written → %s", out_path)
    return out_path

# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def process_retrain_flag(flag_path: Path) -> dict:
    """Handle one retrain flag file end-to-end."""
    with flag_path.open() as fh:
        flag = json.load(fh)

    week_label = flag.get("week", "UNKNOWN")
    logger.info("Processing retrain flag for week: %s", week_label)

    report: dict = {
        "week":           week_label,
        "started_at":     datetime.now().isoformat(),
        "flag_file":      str(flag_path),
        "promoted":       False,
        "skip_reason":    None,
        "prod_roc_auc":   None,
        "new_roc_auc":    None,
        "delta_roc_auc":  None,
    }

    try:
        data            = _load_data()
        prod_roc_auc    = _production_roc_auc(data)
        metrics         = run_full_training(data)
        new_roc_auc     = metrics["candidate_roc_auc"]

        report["prod_roc_auc"]       = prod_roc_auc
        report["new_roc_auc"]        = new_roc_auc
        report["candidate_threshold"]= metrics["candidate_threshold"]

        # ── Promotion decision ──────────────────────────────────────────
        if prod_roc_auc is None:
            promote = True
            reason  = "No production model found — promoting unconditionally."
        else:
            delta   = new_roc_auc - prod_roc_auc
            promote = delta >= MIN_IMPROVEMENT_DELTA
            reason  = (
                f"Δ ROC-AUC = {delta:+.4f} ≥ {MIN_IMPROVEMENT_DELTA} → promote"
                if promote
                else f"Δ ROC-AUC = {delta:+.4f} < {MIN_IMPROVEMENT_DELTA} → keep production"
            )
            report["delta_roc_auc"] = delta

        logger.info(reason)

        if promote:
            _promote_candidate(week_label)
            report["promoted"] = True
        else:
            report["skip_reason"] = reason

        report["finished_at"] = datetime.now().isoformat()
        report["status"]      = "success"

    except Exception as exc:  # noqa: BLE001
        logger.error("Retraining pipeline failed: %s", exc, exc_info=True)
        report["status"]     = "failed"
        report["error"]      = str(exc)
        report["finished_at"]= datetime.now().isoformat()

    _write_report(report, week_label)
    mark_flag_processed(flag_path)
    return report

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Retraining Executor for the IoT Anomaly Detection system."
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            f"Continuously poll {REPORT_DIR} for new retrain flags "
            f"every {WATCH_INTERVAL_SECONDS}s."
        ),
    )
    parser.add_argument(
        "--flag",
        type=str,
        default=None,
        help="Process a specific retrain flag file immediately (bypasses discovery).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    RETRAIN_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.flag:
        # ── Single-shot: process one explicit flag ──────────────────────
        flag_path = Path(args.flag)
        if not flag_path.exists():
            logger.error("Flag file not found: %s", flag_path)
            sys.exit(1)
        process_retrain_flag(flag_path)
        return

    if args.watch:
        # ── Continuous watch loop ───────────────────────────────────────
        logger.info(
            "🔍 Watch mode active. Polling %s every %ds…",
            REPORT_DIR, WATCH_INTERVAL_SECONDS,
        )
        while True:
            flags = find_pending_flags()
            if flags:
                logger.info("Found %d pending retrain flag(s).", len(flags))
                for flag in flags:
                    process_retrain_flag(flag)
            else:
                logger.info("No pending flags. Next check in %ds.", WATCH_INTERVAL_SECONDS)
            time.sleep(WATCH_INTERVAL_SECONDS)
    else:
        # ── Single-shot: process all pending flags found now ───────────
        flags = find_pending_flags()
        if not flags:
            logger.info("No pending retrain flags found in %s.", REPORT_DIR)
            return
        for flag in flags:
            process_retrain_flag(flag)


if __name__ == "__main__":
    main()
