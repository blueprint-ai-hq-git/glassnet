#!/bin/bash
# GRADER RUNNER for P86a-e. CPU only, no model load, parses stored text. Predictions written BEFORE:
# lens/runs/OURS3M-GRADE-PREDICTIONS-P86.txt. Writes ONLY lens/runs/GRADE-P86.txt + GRADE-P86-RUN.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=1
export GRADE_P86_OUT="$PWD/lens/runs/GRADE-P86.txt"
{
  echo "=== GRADE-P86 runner $(date '+%F %H:%M:%S') ==="
  start=$(date +%s)
  nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/grade_ours3m_ep1.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/GRADE-P86-RUN.txt 2>&1
