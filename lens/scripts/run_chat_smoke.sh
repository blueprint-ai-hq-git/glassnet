#!/bin/bash
# run_chat_smoke.sh — verify the chat trainer's code path on CPU before spending GPU time.
# A tiny run: 1 epoch, 2000 windows, cpu. If the ops are wrong it fails here in a minute, not
# three hours into a GPU run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 32 --window 8 --epochs 1 --batch 64 --lr 0.4 --device cpu \
  --max-windows 2000 --out smoke 2>&1 | tail -45
