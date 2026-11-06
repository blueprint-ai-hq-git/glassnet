#!/bin/bash
# PLAN item (f): is the cached decode step overhead-bound, and does a fused step keep decoding IDENTICALLY?
# PREDICTIONS, written before the run: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt
# Output: lens/runs/DECODE-STEP-COST.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== CACHED DECODE STEP: BREAKDOWN + FUSED STEP $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_step_cost.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/DECODE-STEP-COST.txt 2>&1
