"""
Tests: Preprocessing
====================
Validates the IoTPreprocessor's core invariants:
  - Scaler fitted only on normal training sequences
  - Temporal split respects ordering
  - y_train contains ONLY zeros after filtering
"""

import numpy as np
import pytest
from src.data.generator import IoTDataGenerator
from src.data.preprocessor import IoTPreprocessor


@pytest.fixture(scope="module")
def small_dataset():
    """Generate a tiny dataset for fast testing."""
    gen = IoTDataGenerator(n_sensors=10, seq_length=20, n_sequences=200, anomaly_ratio=0.1, seed=0)
    X, y = gen.generate_dataset()
    return X, y


def test_fit_transform_shape(small_dataset):
    X, y = small_dataset
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    X_normal = X[y == 0]
    X_scaled = prep.fit_transform_train(X_normal)
    assert X_scaled.shape == X_normal.shape, "Scaled shape must match input shape"


def test_scaler_mean_close_to_zero(small_dataset):
    """After StandardScaler, reshaped train data should have ~0 mean."""
    X, y = small_dataset
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    X_normal = X[y == 0]
    X_scaled = prep.fit_transform_train(X_normal)
    col_means = X_scaled.reshape(-1, 10).mean(axis=0)
    np.testing.assert_allclose(col_means, 0.0, atol=0.1,
                               err_msg="Scaled train mean should be ~0 per feature")


def test_transform_before_fit_raises():
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    dummy = np.zeros((5, 20, 10), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not been fitted"):
        prep.transform(dummy)


def test_train_contains_only_normals(small_dataset):
    X, y = small_dataset
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    X_tr, y_tr, X_v, y_v, X_te, y_te = prep.train_val_test_split(X, y, train_ratio=0.70)
    assert np.sum(y_tr) == 0, "y_train must contain ONLY normal (0) labels"


def test_temporal_split_ratios(small_dataset):
    X, y = small_dataset
    n = len(X)
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    X_tr, y_tr, X_v, y_v, X_te, y_te = prep.train_val_test_split(X, y, 0.70, 0.15)
    # Val + Test should cover the last 30% — accounting for normal-only filter in train
    assert len(X_v) > 0
    assert len(X_te) > 0
    # Proportions approximate (anomaly removal affects exact counts)
    assert abs(len(X_v) / n - 0.15) < 0.05
    assert abs(len(X_te) / n - 0.15) < 0.05


def test_full_pipeline_saves_files(tmp_path, small_dataset):
    X, y = small_dataset
    prep = IoTPreprocessor(seq_length=20, n_sensors=10)
    splits = prep.full_pipeline(
        X, y,
        save_dir=str(tmp_path / "processed"),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        assert key in splits
        assert (tmp_path / "processed" / f"{key}.npy").exists()
    assert (tmp_path / "scaler.pkl").exists()
