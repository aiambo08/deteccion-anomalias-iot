"""
Drift Pipeline Orchestrator
============================
Orchestrates weekly PSI + KS drift checks and triggers model retraining
when drift criteria are exceeded.

Retraining criteria (OR logic):
  - More than 5 sensors with PSI ≥ 0.25  (severe data drift)
  - More than 10 sensors with KS p < 0.05 (widespread distributional shift)
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

from src.monitoring.psi_monitor import PSIMonitor
from src.monitoring.ks_monitor import KSMonitor
from src.monitoring.alerts import log_drift_report, send_alert


class DriftPipeline:
    """Orchestrate all drift monitoring checks and retrain decisions."""

    RETRAIN_PSI_THRESHOLD = 5   # sensors
    RETRAIN_KS_THRESHOLD  = 10  # sensors

    def __init__(
        self,
        X_baseline: np.ndarray,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        report_dir: str = "logs/drift_reports",
    ) -> None:
        self.psi_monitor = PSIMonitor(X_baseline, psi_threshold)
        self.ks_monitor  = KSMonitor(X_baseline, ks_alpha)
        self.report_dir  = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def weekly_check(
        self,
        X_production: np.ndarray,
        week_label: Optional[str] = None,
    ) -> dict:
        """Run the full weekly drift analysis.

        Args:
            X_production: New data from production (N, T, F).
            week_label:   Human-readable label, e.g. "2026-W10".

        Returns:
            Report dict with PSI results, KS results, and retrain flag.
        """
        if week_label is None:
            week_label = datetime.now().strftime("%Y-W%W")

        print(f"\n{'='*65}")
        print(f"  WEEKLY DRIFT CHECK  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}  [{week_label}]")
        print(f"{'='*65}")

        # ── PSI ─────────────────────────────────────────────────────────
        print("\n[1/2] Running PSI analysis…")
        psi_results, psi_critical = self.psi_monitor.check_all_sensors(X_production)

        # ── KS ──────────────────────────────────────────────────────────
        print("\n[2/2] Running KS test…")
        ks_results = self.ks_monitor.check_all_sensors(X_production)

        # ── Retrain decision ─────────────────────────────────────────────
        n_psi_critical = len(psi_critical)
        n_ks_drifted   = len(ks_results)
        retrain = (
            n_psi_critical >= self.RETRAIN_PSI_THRESHOLD or
            n_ks_drifted   >= self.RETRAIN_KS_THRESHOLD
        )

        print(f"\n{'─'*65}")
        print(f"  PSI critical sensors : {n_psi_critical} / {len(psi_results)}")
        print(f"  KS drifted sensors   : {n_ks_drifted}")
        if retrain:
            print(f"\n  🔄 RETRAINING TRIGGERED  (criteria exceeded)")
            send_alert("RETRAIN", f"Drift detected: {n_psi_critical} PSI critical, "
                       f"{n_ks_drifted} KS drifted. Initiating retraining.")
            self._trigger_retrain(week_label)
        else:
            print("  ✅ No retraining required")

        # ── Persist report ───────────────────────────────────────────────
        report = {
            "week":            week_label,
            "timestamp":       datetime.now().isoformat(),
            "psi_critical":    n_psi_critical,
            "ks_drifted":      n_ks_drifted,
            "retrain":         retrain,
            "psi_results":     psi_results,
            "ks_results":      ks_results,
        }
        log_drift_report(report, report_dir=self.report_dir)

        return report

    def _trigger_retrain(self, week_label: str) -> None:
        """Hook for retraining pipeline (CI/CD, Airflow, etc.)."""
        flag_path = os.path.join(self.report_dir, f"retrain_flag_{week_label}.json")
        with open(flag_path, "w") as f:
            json.dump({"trigger": True, "week": week_label,
                       "timestamp": datetime.now().isoformat()}, f, indent=2)
        print(f"  Retrain flag written → {flag_path}")
        # Extend here: call GitHub Actions webhook, Airflow DAG, etc.
