#!/bin/bash
# TRACK 1 (data/tokenization): vocab 12k -> 24k, ONE change. Recipe == R5 (d256 W32 ep192 asonly).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== building v24000 vocab + caches (CPU) ==="
CD_VOCAB=24000 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/chat_data.py 2>&1 | tail -6
echo "=== training v24000 (GPU) ==="
CD_VOCAB=24000 exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
  --out t7_v24k --max-windows 1500000 2>&1 | tee lens/runs/chat/t7_v24k.txt