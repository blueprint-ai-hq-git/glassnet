#!/bin/bash
# run_code_smoke.sh — verify the code path on CPU before spending GPU time.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
CODE_MAX_PAIRS="${CODE_SMOKE_PAIRS:-4000}" nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 64 --window 16 --epochs 1 --batch 128 --device cpu --max-windows 6000 \
  --n-exec 8 --out codesmoke 2>&1 | tail -45