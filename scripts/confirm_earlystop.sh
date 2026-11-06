#!/bin/bash
# confirm_earlystop.sh — reviewer item (1). CPU, 6 BLAS threads, resume-safe.
#
# T2_MAX_EPOCHS / T2_PATIENCE MUST be exported to match the Step 2 protocol
# (96 / 15). They are read by t2_step2_sweep.py at IMPORT time, so without
# them this script would silently run 48/10 — i.e. it would change two things
# at once (the min-epoch rule AND the max-epochs/patience budget).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t2/confirm
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_MAX_EPOCHS=96
export T2_PATIENCE=15
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3

# refuse to run if the imported budget would not match Step 2
"$PY" - <<'PYEOF' || exit 1
import os, sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
os.environ.setdefault("T2_MAX_EPOCHS", "96")
os.environ.setdefault("T2_PATIENCE", "15")
from t2_step2_sweep import MAX_EPOCHS, PATIENCE
print(f"protocol check: MAX_EPOCHS={MAX_EPOCHS} PATIENCE={PATIENCE}")
if (MAX_EPOCHS, PATIENCE) != (96, 15):
    print("ABORT: budget does not match the Step 2 protocol (96/15)")
    sys.exit(1)
print("protocol check OK")
PYEOF

nice -n 19 ionice -c3 "$PY" scripts/t2_confirm_earlystop.py \
  >> runs/t2/confirm/confirm.log 2>&1
echo "rc=$?" >> runs/t2/confirm/confirm.log