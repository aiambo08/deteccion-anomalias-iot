"""
Run the full test suite using the local virtual environment.
Execute: .venv\Scripts\python.exe run_tests.py
"""
import subprocess, sys, os

venv_python = os.path.join(".venv", "Scripts", "python.exe")
result = subprocess.run(
    [venv_python, "-m", "pytest",
     "tests/test_preprocessing.py",
     "tests/test_features.py",
     "tests/test_kafka.py",
     "tests/test_monitoring.py",
     "tests/test_model.py",
     "-v", "--tb=short"],
)
sys.exit(result.returncode)
