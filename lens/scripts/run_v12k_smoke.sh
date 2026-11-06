#!/bin/bash
# SMOKE for the one-change experiment: best1 recipe + vocab 12000 (1 epoch only).
# Purpose: catch a crash / OOM / shape bug BEFORE committing to a 24-epoch run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
CD_VOCAB=12000 TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 16 --epochs 1 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --device gpu --out v12k_smoke \
  --max-windows 1500000 2>&1 | tail -25
