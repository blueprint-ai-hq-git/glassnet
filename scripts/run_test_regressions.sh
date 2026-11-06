#!/bin/bash
# run_test_regressions.sh — pin the four defects of 2026-10-06.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u tests/test_regressions.py 2>&1 \
  | sed -n '/REGRESSION SUITE/,$p' | tail -22