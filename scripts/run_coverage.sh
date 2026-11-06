#!/bin/bash
# run_coverage.sh — readability coverage of the Target C model.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/t3_coverage.py 2>&1 | sed -n '/READABILITY COVERAGE/,$p' | tail -40