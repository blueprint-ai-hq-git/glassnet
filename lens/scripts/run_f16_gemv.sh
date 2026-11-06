#!/bin/bash
# PLAN item (f2b): OUR OWN FUSED f16 GEMV - the bar. bytes/token, ms/step, token identity, control.
# PREDICTIONS, written before any code was compiled: lens/runs/F16-GEMV-PREDICTIONS.txt (P49a-j).
# Output: lens/runs/F16-GEMV.txt (appended, never overwritten) + lens/runs/f16_gemv.json
# It is QUEUED, not run inline, because this machine has already truncated two training arms at the
# 14 GB ceiling (relgrad05c seed 0 at 6/12 on 2026-10-20, sgd04 seed 1 at 8/12 on 2026-10-21) and the
# watchdog only starts a queue entry when no train_chat is running.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== OUR OWN FUSED f16 GEMV (PLAN item f2b) $(date '+%F %H:%M') ==="
  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/f16_gemv_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/F16-GEMV.txt 2>&1
