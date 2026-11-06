#!/bin/bash
# T10 (talk line): ONE change from the shipping R5 recipe — per-concept pos_weight x3 on the
# two strict-audit-weak concepts (is_a_greeting, is_formal). Everything else identical:
# v12000, d256, W32, lr 0.4 decay 0.5, lam 0.3, seed 0, gpu, assistant-only, 192 ep.
# BAR (pre-registered, binary): strict audit 43/43 AND val_acc >= 30.88 (the R5 ship score).
# Fallback if this fails: T11 = drop the two concepts (41-concept set, strict 41/41 by design).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export TALK_BOOST="is_a_greeting:3,is_formal:3"
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --out t10_talk \
  --max-windows 1500000