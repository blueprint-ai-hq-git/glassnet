#!/bin/bash
# PLAN item (f), third pass: what inside push costs 88-315 us? Run once per thread count: 1 and 3.
# PREDICTIONS, written before the run: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt (ADDENDUM 2, P46n-P46s)
# Output: lens/runs/DECODE-STEP-COST.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
T="${1:-3}"
{
  echo "=== PUSH BREAKDOWN, BLAS THREADS=$T $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_push_breakdown.py "$T"
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/DECODE-STEP-COST.txt 2>&1
