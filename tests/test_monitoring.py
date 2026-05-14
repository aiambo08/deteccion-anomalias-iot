"""
Tests: Monitoring (PSI + KS)
==============================
Validates drift detection math and thresholds.
"""

import numpy as np
import pytest

from src.monitoring.psi_monitor import PSIMonitor, calculate_psi
from src.monitoring.ks_monitor import KSMonitor


@pytest.fixture
def baseline_data():
    rng = np.random.default_rng(0)
    # (1000 samples, 60 timesteps, 10 sensors) — small but realistic shape
    return rng.standard_normal((1000, 60, 10)).astype(np.float32)


# ── PSI ───────────────────────────────────────────────────────────────────────

def test_psi_identical_distributions(baseline_data):
    """PSI between identical distributions should be ≈ 0."""
    flat = baseline_data.reshape(-1, 10)
    psi = calculate_psi(flat[:, 0], flat[:, 0])
    assert psi < 0.01, f"PSI of identical distributions should be ~0, got {psi}"


def test_psi_severe_drift():
    """PSI between very different distributions should exceed 0.25."""
    rng  = np.random.default_rng(1)
    dist1 = rng.standard_normal(2000)
    dist2 = rng.standard_normal(2000) + 5.0   # large shift
    psi = calculate_psi(dist1, dist2)
    assert psi >= 0.25, f"PSI should be ≥ 0.25 for severely shifted distribution, got {psi}"


def test_psi_monitor_no_drift(baseline_data):
    """Production data identical to baseline → no critical sensors."""
    monitor = PSIMonitor(baseline_data, psi_threshold=0.25)
    results, critical = monitor.check_all_sensors(baseline_data)
    assert len(critical) == 0, "No critical drift expected for identical data"


def test_psi_monitor_detects_drift(baseline_data):
    """Production data shifted by 5σ → critical drift detected."""
    rng = np.random.default_rng(2)
    production = baseline_data.copy()
    production[:, :, 0] += 5.0   # shift sensor 0

    monitor = PSIMonitor(baseline_data, psi_threshold=0.25)
    results, critical = monitor.check_all_sensors(production)
    shifted_sensors = [idx for idx, _ in critical]
    assert 0 in shifted_sensors, "Sensor 0 should be in critical list"


def test_psi_monitor_status_labels(baseline_data):
    """verify STATUS_OK appears in results for identical data."""
    monitor = PSIMonitor(baseline_data)
    results, _ = monitor.check_all_sensors(baseline_data)
    statuses = {v["status"] for v in results.values()}
    assert "OK" in statuses


# ── KS ────────────────────────────────────────────────────────────────────────

def test_ks_no_drift_identical(baseline_data):
    """KS on identical distributions → no drifted sensors."""
    monitor = KSMonitor(baseline_data, alpha=0.05)
    drifted = monitor.check_all_sensors(baseline_data)
    assert len(drifted) == 0, "No KS drift expected for identical distributions"


def test_ks_detects_shift(baseline_data):
    """KS should detect a 3σ shift in at least one sensor."""
    production = baseline_data.copy()
    production[:, :, 2] += 3.0   # shift sensor 2

    monitor = KSMonitor(baseline_data, alpha=0.05)
    drifted = monitor.check_all_sensors(production)
    drifted_sensors = [d["sensor"] for d in drifted]
    assert 2 in drifted_sensors, "Sensor 2 should be flagged as drifted"


def test_ks_result_fields(baseline_data):
    """Each KS result must have required keys."""
    production = baseline_data.copy()
    production[:, :, 1] += 4.0

    monitor = KSMonitor(baseline_data, alpha=0.05)
    drifted = monitor.check_all_sensors(production)
    for d in drifted:
        assert "sensor"    in d
        assert "statistic" in d
        assert "p_value"   in d
        assert "severity"  in d
        assert d["severity"] in ("SEVERE", "MODERATE")
