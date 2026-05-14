#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_venv.sh — First-time virtual environment setup (Linux/Mac/WSL)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "[1/3] Creating virtual environment..."
python3 -m venv .venv

echo "[2/3] Activating virtual environment..."
source .venv/bin/activate

echo "[3/3] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
pip install -e . --no-deps

echo ""
echo "✓ Setup complete! Virtual environment ready at .venv/"
echo ""
echo "To activate the venv in the future:"
echo "    source .venv/bin/activate"
echo ""
echo "Quick start:"
echo "    python -m src.data.generator"
echo "    python -m src.data.preprocessor"
echo "    python -m src.features.feature_pipeline"
echo "    python -m src.models.trainer"
echo "    pytest tests/ -v"
echo "    uvicorn src.api.main:app --reload"
