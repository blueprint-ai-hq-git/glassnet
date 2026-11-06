#!/bin/bash
# run_selfgen_smoke.sh — smoke-test the generator on our own model.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/selfgen_smoke.py 2>&1 | sed -n '/SELF-GENERATION SMOKE/,$p' | tail -40