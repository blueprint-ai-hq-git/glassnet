#!/bin/bash
# run_readability_sheet.sh — STEP 0 evidence extraction (no training, no retuning).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1          # import the audit's functions without re-running it
PY=".venv-gpu/bin/python3"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_readability_sheet.py 2>&1