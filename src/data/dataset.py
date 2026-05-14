"""
Dataset Utilities
=================
Helper functions for loading, saving, and inspecting split datasets.
"""

import os
from typing import Dict

import numpy as np
import joblib


def save_splits(splits: Dict[str, np.ndarray], save_dir: str = "data/processed") -> None:
    """Save a dict of named arrays as .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    for name, arr in splits.items():
        path = os.path.join(save_dir, f"{name}.npy")
        np.save(path, arr)
    print(f"  Saved {len(splits)} arrays to {save_dir}/")


def load_splits(save_dir: str = "data/processed") -> Dict[str, np.ndarray]:
    """Load all .npy arrays from a directory into a dict."""
    splits = {}
    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith(".npy"):
            key = fname.replace(".npy", "")
            splits[key] = np.load(os.path.join(save_dir, fname))
    print(f"  Loaded arrays: {list(splits.keys())}")
    return splits


def save_baseline_stats(X_train: np.ndarray, path: str = "data/baseline/X_train_baseline_stats.pkl") -> None:
    """Compute and persist per-feature baseline statistics for drift monitoring."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Reshape (N, T, F) → (N*T, F) for column-wise stats
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    stats = {
        "mean": X_flat.mean(axis=0),
        "std":  X_flat.std(axis=0),
        "p05":  np.percentile(X_flat, 5, axis=0),
        "p25":  np.percentile(X_flat, 25, axis=0),
        "p50":  np.percentile(X_flat, 50, axis=0),
        "p75":  np.percentile(X_flat, 75, axis=0),
        "p95":  np.percentile(X_flat, 95, axis=0),
        "n_samples": len(X_train),
    }
    joblib.dump(stats, path)
    print(f"  Baseline stats saved → {path}")


def load_baseline_stats(path: str = "data/baseline/X_train_baseline_stats.pkl") -> dict:
    return joblib.load(path)


def describe_splits(splits: Dict[str, np.ndarray]) -> None:
    """Print a formatted summary of available splits."""
    print("\n" + "═" * 60)
    print("  DATASET SUMMARY")
    print("═" * 60)
    for name, arr in splits.items():
        print(f"  {name:<15} {str(arr.shape):<25} dtype={arr.dtype}")
    print("═" * 60)
