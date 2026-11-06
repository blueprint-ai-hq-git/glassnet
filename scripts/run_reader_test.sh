#!/bin/bash
# run_reader_test.sh — build the R4 blind cards + the answer key + R2 sensitivity.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_reader_test.py 2>&1 | tail -22