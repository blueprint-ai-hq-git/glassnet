#!/bin/bash
# R3: talk epochs 24 -> 48 (ONE change from R2=t2_d256). d256, W32, lr 0.4 decay 0.5, lam 0.3.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 48 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
  --out t3_ep48 --max-windows 1500000 2>&1 | tee lens/runs/chat/t3_ep48.txt