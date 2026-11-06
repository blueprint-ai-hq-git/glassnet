#!/bin/bash
# diag_lie.sh — run the lie-probe diagnostic at baseline and under supervision.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3A_K=32
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
export T3A_LIE_COL=2
export T3A_BACKEND=cupy
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"

echo "################ BASELINE (lam_concept=0) ################"
T2_LAM_CONCEPT=0 T3A_LR=1.0 nice -n 19 ionice -c3 "$PY" scripts/diag_lie_probe.py 2>&1 | grep -v "ep  *[0-9]"
echo
echo "################ SUPERVISED (lam_concept=1.0) ################"
T2_LAM_CONCEPT=1.0 T3A_LR=0.5 nice -n 19 ionice -c3 "$PY" scripts/diag_lie_probe.py 2>&1 | grep -v "ep  *[0-9]"