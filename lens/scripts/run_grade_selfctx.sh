#!/bin/bash
# THE SELF-CONTEXT ARM'S DECIDER GRADER (P100a-d). Text only: no checkpoint, no model, no GPU, no trainer.
# Predictions: lens/runs/SELFCTX-GRADER-PREDICTIONS-P100.txt (written BEFORE grade_selfctx.py existed).
# Output: lens/runs/GRADE-P98.txt (new file; nothing stored is written to).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/GRADE-P98.txt
{
  echo "===================================================================================================="
  echo "SELF-CONTEXT ARM — DECIDER GRADER (P98e/P98f)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SELFCTX-GRADER-PREDICTIONS-P100.txt (written before this runner existed)"
  echo "text only: no checkpoint opened, no model, no GPU, no trainer started."
  echo "===================================================================================================="
  .venv-gpu/bin/python3 -u lens/scripts/grade_selfctx.py
  echo "GRADER EXIT=$?  $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
tail -n 3 "$OUT"
