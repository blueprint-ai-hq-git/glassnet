#!/bin/bash
# R6: talk window 32 -> 64 (ONE change from R5=t5_ep192). d256, lr 0.4 decay 0.5, lam 0.3, 192 epochs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 64 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
  --out t6_w64 --max-windows 1500000 2>&1 | tee lens/runs/chat/t6_w64.txt