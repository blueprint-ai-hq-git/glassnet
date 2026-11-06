#!/bin/bash
# biasinit.sh — reviewer item (2). Same protocol check as item (1).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t2/biasinit
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_MAX_EPOCHS=96
export T2_PATIENCE=15
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3

"$PY" - <<'PYEOF' || exit 1
import os, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from t2_step2_sweep import MAX_EPOCHS, PATIENCE
print(f"protocol check: MAX_EPOCHS={MAX_EPOCHS} PATIENCE={PATIENCE}")
if (MAX_EPOCHS, PATIENCE) != (96, 15):
    print("ABORT: budget does not match the Step 2 / item-(1) protocol (96/15)")
    sys.exit(1)
print("protocol check OK")
PYEOF

nice -n 19 ionice -c3 "$PY" scripts/t2_bias_init.py \
  >> runs/t2/biasinit/biasinit.log 2>&1
echo "rc=$?" >> runs/t2/biasinit/biasinit.log