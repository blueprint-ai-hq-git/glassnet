#!/bin/bash
# run_l3_gate.sh — the L3 gate.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/tests/test_l3.py 2>&1 \
  | sed -n '/L3 GATE/,$p' | tail -36