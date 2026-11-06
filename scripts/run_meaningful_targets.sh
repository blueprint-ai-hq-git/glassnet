#!/bin/bash
# run_meaningful_targets.sh — rung 2 target derivation (no training).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_meaningful_targets.py 2>&1 | sed -n '/RUNG 2/,$p'