#!/bin/bash
# Can the server serve an ATTENTION ship on the GPU, or does the KV cache raise? (found during PLAN item (f))
# Output: lens/runs/GPU-ATTN-CACHE.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== GPU ATTENTION SHIP + KV CACHE $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/gpu_attn_cache_probe.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/GPU-ATTN-CACHE.txt 2>&1
