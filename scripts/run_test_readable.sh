#!/bin/bash
# run_test_readable.sh — scenario suite for today's readable-model modules.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export T3M_IMPORT_ONLY=1
export T3M_K=9
nice -n 19 ionice -c3 "/home/yahya/.hermes/hermes-agent/venv/bin/python3" -u \
  tests/test_readable.py 2>&1 | sed -n '/READABLE-MODEL SCENARIO SUITE/,$p' | tail -34