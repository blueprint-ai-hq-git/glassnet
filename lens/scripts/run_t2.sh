#!/bin/bash
# T2: talk width d 128 -> 256 (ONE change from the new best baseline t1_w32_ep24, W=32).
# Same data, vocab 12k, lr 0.4 decay 0.5, lam-concept 0.3, 24 epochs, seed 0, assistant-only.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
  --out t2_d256 --max-windows 1500000 2>&1 | tee lens/runs/chat/t2_d256.txt