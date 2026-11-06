#!/bin/bash
# run_more_targets.sh — extended concept derivation (no training).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/t3_more_targets.py 2>&1 | sed -n '/MORE MEANINGFUL/,$p' | tail -45