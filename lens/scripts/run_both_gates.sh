#!/bin/bash
# run_both_gates.sh — rebuild lens, then run the L0 and L1 gates in order.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"

echo "=== rebuild ==="
bash lens/scripts/build_lens.sh 2>&1 | tail -2
echo
echo "=== L0 gate (kernels) ==="
nice -n 19 ionice -c3 "$PY" -u lens/tests/test_l0.py 2>&1 | sed -n '/L0 GATE/,$p' | tail -12
echo
echo "=== L1 gate (autograd) ==="
nice -n 19 ionice -c3 "$PY" -u lens/tests/test_l1.py 2>&1 | sed -n '/L1 GATE/,$p' | tail -26