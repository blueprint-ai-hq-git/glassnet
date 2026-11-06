#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "=== SMOKE: 1 seed, 80/class, 2 epochs ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/eyes56.py \
  --n-per 80 --epochs 2 --seeds 0 --device cpu 2>&1 | tail -20
