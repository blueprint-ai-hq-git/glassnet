#!/bin/bash
# Does the KV cache decode IDENTICALLY and FASTER once wired into the decoder and the server?
# PLAN item (e). PREDICTIONS, written before the run: lens/runs/KV-DECODE-PREDICTIONS.txt
# Output: lens/runs/KV-DECODE.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== KV CACHE WIRED INTO DECODING $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/kv_decode_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/KV-DECODE.txt 2>&1
