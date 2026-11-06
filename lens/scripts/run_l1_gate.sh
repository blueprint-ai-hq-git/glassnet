#!/bin/bash
# run_l1_gate.sh — the L1 gate for lens (our own autograd).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/tests/test_l1.py 2>&1 \
  | sed -n '/L1 GATE/,$p' | tail -30