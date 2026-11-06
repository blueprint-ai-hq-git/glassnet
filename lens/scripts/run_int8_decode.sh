#!/bin/bash
# PLAN item (f2): do int8 weights make the cached decode step cheaper, and do they change the answer?
# PREDICTIONS, written before the run: lens/runs/INT8-DECODE-PREDICTIONS.txt (P47a-h)
# Output: lens/runs/INT8-DECODE.txt (appended, never overwritten) + lens/runs/int8_decode.json
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== INT8 WEIGHTS FOR THE DECODE PATH $(date '+%F %H:%M') ==="
  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/int8_decode_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/INT8-DECODE.txt 2>&1
