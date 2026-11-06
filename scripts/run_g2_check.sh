#!/bin/bash
# run_g2_check.sh — is the concept loss actually applied? Run both ways and compare.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"

echo "### A) WITH T3M_IMPORT_ONLY=1 (what the G2 sweep used)"
CHK_MODE=import_only nice -n 19 ionice -c3 "$PY" -u scripts/g2_check_supervision.py 2>&1 | tail -12
echo
echo "### B) WITHOUT T3M_IMPORT_ONLY (what Target E used)"
CHK_MODE=normal nice -n 19 ionice -c3 "$PY" -u scripts/g2_check_supervision.py 2>&1 | tail -12