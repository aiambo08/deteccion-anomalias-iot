"""
Run the full test suite using the uv-managed virtual environment.

Usage:
    # With the venv activated (uv or standard):
    python run_tests.py

    # Or directly via uv (no activation required):
    uv run python run_tests.py
"""

import os
import subprocess
import sys

# ── Resolve the Python interpreter ───────────────────────────────────────────
# Prefer the venv interpreter so the test environment exactly matches the one
# created by setup_venv.{sh,bat} / `uv venv`.  Fall back to the current
# interpreter (useful inside CI where the venv *is* the system Python, or
# when running via `uv run`).

def _venv_python() -> str:
    """Return the path to the venv Python, or sys.executable as a fallback."""
    candidates = [
        os.path.join(".venv", "Scripts", "python.exe"),  # Windows
        os.path.join(".venv", "bin", "python"),           # Linux / macOS / WSL
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # No .venv found — use the interpreter that is running this script.
    # This is the correct behaviour when using `uv run` or inside Docker.
    return sys.executable


TEST_MODULES = [
    "tests/test_preprocessing.py",
    "tests/test_features.py",
    "tests/test_kafka.py",
    "tests/test_monitoring.py",
    "tests/test_model.py",
    "tests/test_api.py",
]

if __name__ == "__main__":
    python = _venv_python()
    print(f"Running tests with: {python}\n")

    result = subprocess.run(
        [python, "-m", "pytest", *TEST_MODULES, "-v", "--tb=short"],
    )
    sys.exit(result.returncode)
