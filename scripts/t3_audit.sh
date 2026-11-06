#!/bin/bash
# t3_audit.sh — reviewer item (1): concept audit for the K=64 lr=0.3 readable,
# 3 seeds + the lie probe. GPU for training, host numpy for the audit maths.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3A_BACKEND=cupy
export T3A_K=64
export T3A_LR=0.3
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "=== item (1) audit  $(date '+%F %T') ===" >> runs/t3/audit.log
nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
echo "audit rc=$?" >> runs/t3/audit.log