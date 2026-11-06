#!/bin/bash
# c_category_test.sh — diagnose R2 = 0/6 (category ablation vs random control).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/t3_c_category_test.py 2>&1 | tail -22