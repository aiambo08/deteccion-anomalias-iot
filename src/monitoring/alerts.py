"""
Alert System
============
Alert formatting and logging utilities for drift and anomaly events.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def send_alert(level: str, message: str) -> None:
    """Log and optionally dispatch an alert.

    Args:
        level:   Alert severity string (e.g. "RETRAIN", "HIGH", "MEDIUM").
        message: Human-readable alert message.
    """
    ts = datetime.now().isoformat()
    log_line = f"[{ts}] [{level}] {message}"

    # Console output
    print(f"\n  🔔 ALERT  [{level}]  {message}")
    logger.warning(log_line)

    # Persist to log file
    os.makedirs("logs/anomaly_alerts", exist_ok=True)
    log_path = f"logs/anomaly_alerts/alerts_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_drift_report(report: dict, report_dir: str = "logs/drift_reports") -> str:
    """Persist a drift report dict as a JSON file.

    Args:
        report:     Dict containing drift check results.
        report_dir: Directory to write reports.

    Returns:
        Path to the saved report file.
    """
    os.makedirs(report_dir, exist_ok=True)
    fname = f"drift_{report.get('week', 'unknown')}_{datetime.now().strftime('%H%M%S')}.json"
    path  = os.path.join(report_dir, fname)

    # PSI results may contain non-serialisable numpy floats — normalise
    def _serialise(obj: Any):
        if isinstance(obj, (float, int)):
            return obj
        return str(obj)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_serialise)

    print(f"  Drift report saved → {path}")
    return path
