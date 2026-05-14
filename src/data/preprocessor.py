"""
IoT Preprocessor
================
Handles data cleaning, normalisation, and train/val/test splitting
with strict temporal ordering to avoid data leakage.

Key design decisions:
  - StandardScaler fitted *only* on normal training sequences
  - Temporal (no-shuffle) 70/15/15 split
  - X_train filtered to normal-only sequences (critical for autoencoder)
"""

import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


class IoTPreprocessor:
    """Normalisation and splitting for multivariate IoT sequences."""

    def __init__(self, seq_length: int = 60, n_sensors: int = 50) -> None:
        self.seq_length = seq_length
        self.n_sensors = n_sensors
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def fit_transform_train(self, X_train_raw: np.ndarray) -> np.ndarray:
        """Fit scaler on train data and return normalised sequences.

        IMPORTANT: Must be called ONLY with normal training sequences.
        Reshapes (N, T, F) → (N*T, F) for fitting, then reshapes back.
        """
        n_samples, seq_len, n_feat = X_train_raw.shape
        X_flat = X_train_raw.reshape(-1, n_feat)
        X_scaled = self.scaler.fit_transform(X_flat)
        self._fitted = True
        return X_scaled.reshape(n_samples, seq_len, n_feat).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the already-fitted scaler to val/test/production data."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_transform_train first.")
        n_samples, seq_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(n_samples, seq_len, n_feat).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scaler(self, path: str = "models/scaler.pkl") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"  Scaler saved → {path}")

    def load_scaler(self, path: str = "models/scaler.pkl") -> None:
        self.scaler = joblib.load(path)
        self._fitted = True
        print(f"  Scaler loaded ← {path}")

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Temporal (ordered) split — NO shuffling to avoid leakage.

        X_train is filtered to contain ONLY normal sequences.
        Val and Test preserve the natural mix of normal + anomalies.

        Returns:
            X_train_clean, y_train_clean, X_val, y_val, X_test, y_test
        """
        n = len(X)
        i_train = int(n * train_ratio)
        i_val = int(n * (train_ratio + val_ratio))

        X_train_all, y_train_all = X[:i_train], y[:i_train]
        X_val, y_val = X[i_train:i_val], y[i_train:i_val]
        X_test, y_test = X[i_val:], y[i_val:]

        # CRITICAL: Keep only normals in train for autoencoder
        normal_mask = y_train_all == 0
        X_train_clean = X_train_all[normal_mask]
        y_train_clean = y_train_all[normal_mask]

        print("═" * 55)
        print("  DATA SPLIT SUMMARY")
        print("═" * 55)
        print(f"  Train (normals only): {X_train_clean.shape}"
              f"  ({y_train_clean.sum()} anomalies — must be 0)")
        print(f"  Val   (mixed)       : {X_val.shape}"
              f"  ({y_val.sum()} anomalies, {y_val.mean()*100:.1f}%)")
        print(f"  Test  (mixed)       : {X_test.shape}"
              f"  ({y_test.sum()} anomalies, {y_test.mean()*100:.1f}%)")

        assert y_train_clean.sum() == 0, (
            "FAIL: y_train_clean contains anomalies! Check filtering logic."
        )

        return X_train_clean, y_train_clean, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # End-to-end pipeline helper
    # ------------------------------------------------------------------

    def full_pipeline(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        save_dir: str = "data/processed",
        scaler_path: str = "models/scaler.pkl",
    ) -> dict:
        """Run the complete preprocessing pipeline and save artefacts.

        Steps:
          1. Temporal split (raw, unscaled)
          2. Fit scaler on X_train_clean (normal only)
          3. Scale all splits
          4. Save .npy arrays and scaler

        Returns dict with all split arrays.
        """
        print("\n[1] Splitting data…")
        X_tr, y_tr, X_v, y_v, X_te, y_te = self.train_val_test_split(X_raw, y_raw)

        print("\n[2] Fitting scaler on normal training sequences…")
        X_tr_scaled = self.fit_transform_train(X_tr)

        print("[3] Transforming val/test…")
        X_v_scaled = self.transform(X_v)
        X_te_scaled = self.transform(X_te)

        print("\n[4] Saving artefacts…")
        os.makedirs(save_dir, exist_ok=True)
        splits = {
            "X_train": X_tr_scaled, "y_train": y_tr,
            "X_val":   X_v_scaled,  "y_val":   y_v,
            "X_test":  X_te_scaled, "y_test":  y_te,
        }
        for name, arr in splits.items():
            np.save(os.path.join(save_dir, f"{name}.npy"), arr)
            print(f"    {name}.npy  {arr.shape}")

        self.save_scaler(scaler_path)

        print("\n  Preprocessing complete ✓")
        return splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_raw = np.load("data/raw/X.npy")
    y_raw = np.load("data/raw/y.npy")

    prep = IoTPreprocessor()
    splits = prep.full_pipeline(X_raw, y_raw)
