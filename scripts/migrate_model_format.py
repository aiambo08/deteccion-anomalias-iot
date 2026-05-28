"""
Migrate Legacy Model Artefact to Keras 3 Native Format
=======================================================
The legacy HDF5 format (.h5) stores metric/loss names as strings
(e.g. 'keras.metrics.mse') that Keras 3 can no longer deserialise.

This one-time script:
  1. Loads the .h5 model with compile=False (skips broken metric deserialisation)
  2. Recompiles with identical settings (adam + mse)
  3. Saves to the native .keras format (Keras 3 default)
  4. Keeps the original .h5 as a backup (.h5.bak)

After running this script all loaders (InferenceEngine, consumer) will
find a .keras file which loads cleanly without compile=False workarounds.

Usage:
    .venv\\Scripts\\python.exe scripts/migrate_model_format.py
    .venv\\Scripts\\python.exe scripts/migrate_model_format.py --src models/bilstm_autoencoder.h5
"""

import argparse
import shutil
import sys
from pathlib import Path


def migrate(src: Path, dst: Path) -> None:
    import tensorflow as tf  # deferred — TF startup is slow

    print(f"\n  Source : {src}")
    print(f"  Target : {dst}")

    if not src.exists():
        print(f"\n  ERROR: Source model not found: {src}", file=sys.stderr)
        sys.exit(1)

    if dst.exists():
        print(f"\n  Target already exists — skipping migration.")
        print("  Delete the target file manually if you want to re-run.")
        return

    # ── Step 1: load without compiling ──────────────────────────────────
    print("\n  [1/4] Loading legacy model (compile=False)…")
    model = tf.keras.models.load_model(str(src), compile=False)
    print(f"        Loaded OK  params={model.count_params():,}")

    # ── Step 2: recompile ───────────────────────────────────────────────
    print("  [2/4] Recompiling (adam + mse)…")
    model.compile(optimizer="adam", loss="mse")

    # ── Step 3: save in native .keras format ────────────────────────────
    print(f"  [3/4] Saving to {dst} …")
    dst.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dst))
    print("        Saved OK")

    # ── Step 4: backup original ─────────────────────────────────────────
    bak = src.with_suffix(".h5.bak")
    print(f"  [4/4] Backing up original -> {bak}")
    shutil.copy2(src, bak)

    print("\n  ✅  Migration complete.")
    print(f"       New model: {dst}")
    print(f"       Backup   : {bak}")
    print("\n  The existing loaders have a compile=False fallback so both")
    print("  formats work. Once you confirm the new model works correctly")
    print("  you can delete the .h5 and .h5.bak files.")


def _verify(path: Path) -> None:
    """Quick sanity check: load the migrated model and run a dummy prediction."""
    import numpy as np
    import tensorflow as tf

    print(f"\n  Verifying {path} …")
    model = tf.keras.models.load_model(str(path))
    dummy = np.zeros((1, 60, 50), dtype="float32")
    out = model.predict(dummy, verbose=0)
    assert out.shape == (1, 60, 50), f"Unexpected output shape: {out.shape}"
    print("  ✅  Verification passed — output shape (1, 60, 50) OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate .h5 model to .keras format")
    parser.add_argument(
        "--src",
        default="models/bilstm_autoencoder.h5",
        help="Path to the legacy .h5 model (default: models/bilstm_autoencoder.h5)",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="Output path for the .keras model (default: same dir, .keras extension)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run a quick prediction after migration to verify (default: True)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_suffix(".keras")

    migrate(src, dst)

    if args.verify and dst.exists():
        _verify(dst)
