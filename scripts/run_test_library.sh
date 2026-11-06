#!/bin/bash
# run_test_library.sh — the library scenario suite.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u tests/test_library.py 2>&1 | tail -45