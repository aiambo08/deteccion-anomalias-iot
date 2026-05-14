@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: setup_venv.bat — First-time virtual environment setup
:: Run ONCE after cloning the repository.
:: ─────────────────────────────────────────────────────────────────────────────

echo [1/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: python -m venv failed. Make sure Python 3.10+ is installed.
    exit /b 1
)

echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt
pip install -e . --no-deps

echo.
echo ✓ Setup complete! Virtual environment ready at .venv\
echo.
echo To activate the venv in the future, run:
echo     .venv\Scripts\activate.bat
echo.
echo Quick start:
echo     python -m src.data.generator
echo     python -m src.data.preprocessor
echo     python -m src.features.feature_pipeline
echo     python -m src.models.trainer
echo     pytest tests\ -v
echo     uvicorn src.api.main:app --reload
