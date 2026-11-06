#!/bin/bash
# TRACK 2 (architecture): causal self-attention before the concept layer.
# ONE change from R5 (shipped talk, 30.88%): --attention. Same v12k data, d256, W32, 192 epochs.
# GATE (pre-registered): vs t5_ep192 (30.88%) bar = 32.88% + unk<=1% + fluent -> else FAIL.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
CD_VOCAB=12000 exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --attention \
  --out t9_attn --max-windows 1500000 2>&1 | tee lens/runs/chat/t9_attn.txt