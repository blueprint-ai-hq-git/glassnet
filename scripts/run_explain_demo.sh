#!/bin/bash
# run_explain_demo.sh — glassnet explain() demo + the exactness proof.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/explain_demo.py 2>&1 | sed -n '/GLASSNET explain/,$p' | tail -60