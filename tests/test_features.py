"""
Tests: Feature Engineering
===========================
Validates output shapes and basic numerical properties of each feature module.
"""

import numpy as np
import pytest

from src.features.delta_features import compute_delta_features
from src.features.fft_features import compute_fft_features
from src.features.lag_features import compute_lag_features
from src.features.feature_pipeline import extract_advanced_features, build_feature_matrix, FEATURE_DIM


SEQ_LEN   = 60
N_SENSORS = 50


@pytest.fixture
def sample_sequence():
    rng = np.random.default_rng(42)
    return rng.standard_normal((SEQ_LEN, N_SENSORS)).astype(np.float32)


@pytest.fixture
def batch_sequences():
    rng = np.random.default_rng(0)
    return rng.standard_normal((100, SEQ_LEN, N_SENSORS)).astype(np.float32)


# ── Delta features ────────────────────────────────────────────────────────────

def test_delta_shape(sample_sequence):
    out = compute_delta_features(sample_sequence)
    assert out.shape == (N_SENSORS * 5,), f"Expected ({N_SENSORS * 5},), got {out.shape}"


def test_delta_dtype(sample_sequence):
    out = compute_delta_features(sample_sequence)
    assert out.dtype == np.float32


def test_delta_finite(sample_sequence):
    out = compute_delta_features(sample_sequence)
    assert np.all(np.isfinite(out)), "Delta features should be finite"


# ── FFT features ──────────────────────────────────────────────────────────────

def test_fft_shape(sample_sequence):
    out = compute_fft_features(sample_sequence)
    assert out.shape == (N_SENSORS * 9,), f"Expected ({N_SENSORS * 9},), got {out.shape}"


def test_fft_spectral_entropy_positive(sample_sequence):
    out = compute_fft_features(sample_sequence)
    # Spectral entropy is at position base+8 for each sensor
    entropy_indices = [i * 9 + 8 for i in range(N_SENSORS)]
    entropies = out[entropy_indices]
    assert np.all(entropies >= 0), "Spectral entropy must be non-negative"


# ── Lag features ──────────────────────────────────────────────────────────────

def test_lag_shape(sample_sequence):
    out = compute_lag_features(sample_sequence)
    assert out.shape == (N_SENSORS * 7,), f"Expected ({N_SENSORS * 7},), got {out.shape}"


def test_lag_autocorr_range(sample_sequence):
    out = compute_lag_features(sample_sequence)
    # Autocorr values (positions 0,1,2 per sensor block) should be in [-1, 1]
    for i in range(N_SENSORS):
        base = i * 7
        for offset in range(3):
            val = out[base + offset]
            assert -1.01 <= val <= 1.01, f"Autocorr out of range: {val}"


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_pipeline_dim(sample_sequence):
    out = extract_advanced_features(sample_sequence)
    assert out.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {out.shape}"


def test_build_feature_matrix_shape(batch_sequences):
    out = build_feature_matrix(batch_sequences, verbose=False)
    assert out.shape == (100, FEATURE_DIM)


def test_pipeline_no_nan(sample_sequence):
    out = extract_advanced_features(sample_sequence)
    assert not np.any(np.isnan(out)), "Pipeline output must not contain NaN"


def test_anomaly_features_differ_from_normal():
    """Anomaly sequences should produce different feature vectors from normal ones."""
    from src.data.generator import IoTDataGenerator
    gen = IoTDataGenerator(n_sensors=N_SENSORS, seq_length=SEQ_LEN, seed=1)
    normal_seq   = gen.generate_normal_sequence()
    anomaly_seq  = gen.inject_anomaly(normal_seq, anomaly_type="spike")

    f_normal  = extract_advanced_features(normal_seq)
    f_anomaly = extract_advanced_features(anomaly_seq)

    # At least some features should differ
    diff = np.abs(f_normal - f_anomaly)
    assert np.max(diff) > 0.01, "Anomaly features should differ from normal features"
