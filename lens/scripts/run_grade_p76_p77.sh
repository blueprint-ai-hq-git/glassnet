#!/bin/bash
# P80: grade the two runs whose predictions were never read (P76 and P77), from the stored files only.
# CPU only, no GPU, no checkpoint opened, no model loaded. Writes lens/runs/GRADE-P76-P77.txt (new file).
# Predictions were written BEFORE this runner existed: lens/runs/GRADE-P76-P77-PREDICTIONS-P80.txt
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== GRADE P76/P77 $(date '+%F %H:%M') ==="
  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/grade_p76_p77.py
  echo "python exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/GRADE-P76-P77-RUN.txt 2>&1
