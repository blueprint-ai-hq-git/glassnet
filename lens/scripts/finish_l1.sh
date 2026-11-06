#!/bin/bash
# finish_l1.sh — re-run the L1 gate to confirm, then record.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY=".venv-gpu/bin/python3"

echo "=== L1 gate (final) ==="
nice -n 19 ionice -c3 "$PY" -u lens/tests/test_l1.py 2>&1 | sed -n '/L1 GATE/,$p' | tail -24
echo
echo "=== record ==="
"$PY" -u lens/scripts/record_l1.py 2>&1 | tail -3
cp STATUS.md REBUILD-PLAN.md /home/yahya/ 2>/dev/null
echo "copied to ~"