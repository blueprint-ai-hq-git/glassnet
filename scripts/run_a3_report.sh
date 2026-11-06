#!/bin/bash
# run_a3_report.sh — TARGET A3 evaluation: four clauses, hold-out, margin, per-column.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_a3_report.py 2>&1