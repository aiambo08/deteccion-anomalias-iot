"""
Feature Pipeline
================
Combines delta, FFT, and lag features into a unified 1050-dimensional
feature vector per sequence. Used by both XGBoost training and the
real-time Kafka consumer.

Total features:
  Delta   :  50 sensors × 5  =  250
  FFT     :  50 sensors × 9  =  450
  Lag     :  50 sensors × 7  =  350
             ─────────────────────
  Total   :                  = 1050
"""

import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.features.delta_features import compute_delta_features
from src.features.fft_features import compute_fft_features
from src.features.lag_features import compute_lag_features

FEATURE_DIM = 1050  # 250 + 450 + 350


def extract_advanced_features(sequence: np.ndarray) -> np.ndarray:
    """Compute the full 1050-dim feature vector for one sequence.

    Args:
        sequence: (T, F) array — e.g. (60, 50).

    Returns:
        features: (1050,) float32 array.
    """
    delta = compute_delta_features(sequence)   # (250,)
    fft   = compute_fft_features(sequence)     # (450,)
    lag   = compute_lag_features(sequence)     # (350,)
    combined = np.concatenate([delta, fft, lag])
    assert combined.shape[0] == FEATURE_DIM, (
        f"Feature dim mismatch: expected {FEATURE_DIM}, got {combined.shape[0]}"
    )
    return combined.astype(np.float32)


def build_feature_matrix(
    X: np.ndarray,
    verbose: bool = True,
    save_path: Optional[str] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Build the full feature matrix for a dataset split.

    Args:
        X: (N, T, F) sequence array.
        verbose: Whether to show a tqdm progress bar (only in single-job mode).
        save_path: If provided, save the resulting matrix as .npy.
        n_jobs: Number of parallel jobs for feature extraction.
            ``1``  = sequential (default, safe everywhere).
            ``-1`` = use all available CPU cores (recommended for large datasets).

    Returns:
        features: (N, 1050) float32 array.
    """
    n = len(X)

    if n_jobs != 1:
        # Parallel extraction — faster for large N, but disables progress bar
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(extract_advanced_features)(X[i]) for i in range(n)
        )
        features = np.array(results, dtype=np.float32)
    else:
        features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        iterator = tqdm(range(n), desc="Extracting features") if verbose else range(n)
        for i in iterator:
            features[i] = extract_advanced_features(X[i])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, features)
        print(f"  Features saved → {save_path}  {features.shape}")

    return features



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.dataset import load_splits

    splits = load_splits("data/processed")

    print("\nBuilding training features…")
    X_train_adv = build_feature_matrix(
        splits["X_train"], save_path="data/features/X_train_advanced.npy"
    )
    print("\nBuilding validation features…")
    X_val_adv = build_feature_matrix(
        splits["X_val"], save_path="data/features/X_val_advanced.npy"
    )
    print("\nBuilding test features…")
    X_test_adv = build_feature_matrix(
        splits["X_test"], save_path="data/features/X_test_advanced.npy"
    )

    print(f"\nFeature matrix shapes:")
    print(f"  Train : {X_train_adv.shape}")
    print(f"  Val   : {X_val_adv.shape}")
    print(f"  Test  : {X_test_adv.shape}")
