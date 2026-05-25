#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_venv.sh — First-time virtual environment setup (Linux / macOS / WSL)
#
# Uses `uv` (https://github.com/astral-sh/uv) for ~10-100× faster installs
# and fully reproducible environments via uv.lock.
#
# Run ONCE after cloning the repository.
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── 0. Ensure uv is available ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[0/4] uv not found — installing via the official installer…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in the current shell session
    export PATH="$HOME/.local/bin:$PATH"
    echo "      ✓ uv installed at $(which uv)"
else
    echo "[0/4] uv already installed: $(uv --version)"
fi

# ── 1. Create virtual environment ─────────────────────────────────────────────
echo "[1/4] Creating virtual environment with uv…"
uv venv .venv --python 3.10

# ── 2. Activate ───────────────────────────────────────────────────────────────
echo "[2/4] Activating virtual environment…"
source .venv/bin/activate

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing dependencies from requirements.txt…"
uv pip install -r requirements.txt

# ── 4. Editable install ───────────────────────────────────────────────────────
echo "[4/4] Installing project in editable mode (src.* imports)…"
uv pip install -e . --no-deps

echo ""
echo "✓ Setup complete! Virtual environment ready at .venv/"
echo ""
echo "To activate the venv in future sessions:"
echo "    source .venv/bin/activate"
echo ""
echo "Quick start:"
echo "    python -m src.data.generator"
echo "    python -m src.data.preprocessor"
echo "    python -m src.features.feature_pipeline"
echo "    python -m src.models.trainer"
echo "    pytest tests/ -v"
echo "    uvicorn src.api.main:app --reload"
