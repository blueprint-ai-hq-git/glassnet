#!/bin/bash
# rebuild_and_gate.sh — rebuild lens after a kernel fix and re-run the L0 gate.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
echo "=== rebuild ==="
bash lens/scripts/build_lens.sh 2>&1 | tail -3
echo
echo "=== L0 gate ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/tests/test_l0.py 2>&1 \
  | sed -n '/L0 GATE/,$p' | tail -34