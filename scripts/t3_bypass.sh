#!/bin/bash
# t3_bypass.sh — step 3: is anything bypassing the concept layer?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export BYP_BACKEND=cupy
export BYP_K=64
export BYP_LR=0.3
export BYP_SEED=0
export BYP_EPOCHS=96
export BYP_MIN_EPOCH=30
export BYP_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "=== step 3 bypass check  $(date '+%F %T') ===" >> runs/t3/bypass.log
nice -n 19 ionice -c3 "$PY" scripts/t3_bypass.py >> runs/t3/bypass.log 2>&1
echo "bypass rc=$?" >> runs/t3/bypass.log