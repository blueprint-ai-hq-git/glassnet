#!/bin/bash
# run_char_smoke.sh — verify the character-level path on CPU before spending GPU time.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat_char.py \
  --d 32 --window 32 --epochs 1 --batch 128 --device cpu --max-windows 8000 --out charsmoke 2>&1 | tail -45