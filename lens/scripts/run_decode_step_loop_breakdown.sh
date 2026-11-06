#!/bin/bash
# PLAN item (f), second pass: the breakdown must ADD UP - push and the step timed INSIDE the loop.
# PREDICTIONS, written before the run: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt (ADDENDUM 1, P46i-P46m)
# Output: lens/runs/DECODE-STEP-COST.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== CACHED DECODE STEP: IN-LOOP BREAKDOWN + THE SHIFT $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_step_loop_breakdown.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/DECODE-STEP-COST.txt 2>&1
