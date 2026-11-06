#!/bin/bash
# run_g1_probe.sh — measure why two concepts fail at every size.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g1_probe.py 2>&1 \
  | sed -n '/WHY DO TWO CONCEPTS FAIL/,$p' | tail -32