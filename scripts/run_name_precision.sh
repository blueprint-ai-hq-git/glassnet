#!/bin/bash
# run_name_precision.sh — METER-04 R2 on existing checkpoints (no training).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_name_precision.py 2>&1 | sed -n '/METER-04 R2/,$p'