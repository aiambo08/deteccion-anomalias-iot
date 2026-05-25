@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: setup_venv.bat — First-time virtual environment setup (Windows)
::
:: Uses `uv` (https://github.com/astral-sh/uv) for ~10-100x faster installs
:: and fully reproducible environments via uv.lock.
::
:: Run ONCE after cloning the repository.
:: ─────────────────────────────────────────────────────────────────────────────

:: ── 0. Ensure uv is available ────────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [0/4] uv not found -- installing via the official Windows installer...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: uv installation failed. Install manually from https://github.com/astral-sh/uv/releases
        exit /b 1
    )
    echo       uv installed successfully.
) else (
    for /f "tokens=*" %%v in ('uv --version') do echo [0/4] uv already installed: %%v
)

:: ── 1. Create virtual environment ────────────────────────────────────────────
echo [1/4] Creating virtual environment with uv...
uv venv .venv --python 3.10
if errorlevel 1 (
    echo ERROR: uv venv failed. Make sure Python 3.10+ is installed and on PATH.
    exit /b 1
)

:: ── 2. Activate ──────────────────────────────────────────────────────────────
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: ── 3. Install dependencies ──────────────────────────────────────────────────
echo [3/4] Installing dependencies from requirements.txt...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)

:: ── 4. Editable install ──────────────────────────────────────────────────────
echo [4/4] Installing project in editable mode (src.* imports)...
uv pip install -e . --no-deps
if errorlevel 1 (
    echo ERROR: Editable install failed.
    exit /b 1
)

echo.
echo √ Setup complete! Virtual environment ready at .venv\
echo.
echo To activate the venv in future sessions:
echo     .venv\Scripts\activate.bat
echo.
echo Quick start:
echo     python -m src.data.generator
echo     python -m src.data.preprocessor
echo     python -m src.features.feature_pipeline
echo     python -m src.models.trainer
echo     pytest tests\ -v
echo     uvicorn src.api.main:app --reload
