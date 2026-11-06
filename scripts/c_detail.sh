#!/bin/bash
# c_detail.sh — Target C per-column audit + R2 name precision + R4 cards.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/t3_c_detail.py 2>&1 | tail -40