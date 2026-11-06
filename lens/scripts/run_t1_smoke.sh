#!/bin/bash
# T1: talk window 16 -> 32 (ONE change), assistant-only, everything else identical to
# asonly_v12000_fixedgrad (d=128, vocab 12k, 24ep, lr 0.4 decay 0.5, lam 0.3).
# SMOKE PHASE: 4 epochs to prove health + direction before the full run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 32 --epochs 4 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only \
  --out t1_w32_smoke --max-windows 1500000 2>&1 | tee lens/runs/chat/t1_w32_smoke.txt