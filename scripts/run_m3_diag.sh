#!/bin/bash
# run_m3_diag.sh — target-occurrence + live-only rejection diagnostics (re-score).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_m3_diag.py 2>&1