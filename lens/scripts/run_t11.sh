#!/bin/bash
# T11 (talk line, final): ONE change from the T10 run — concept set 43 -> 41 (DROP
# is_a_greeting + is_formal). Everything else identical (v12000, d256, W32, lr 0.4 decay 0.5,
# lam 0.3, seed 0, gpu, assistant-only, 192 ep; NO boost).
# WHY DROP (measured, 2026-10-11): both weak audits trace to corpus rarity — greeting phrases
# ("good morning", "how are you") are not even IN the v12000 vocab; support floor 0.5-0.7% of
# windows; x3 pos_weight changed nothing (false-proofed on the right subject this time). The
# remaining 41 concepts are honest in every prior measurement, so strict 41/41 is expected.
# BAR (pre-registered, binary): strict audit 41/41 AND val_acc >= 30.88 (the R5 ship score).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --out t11_talk41 \
  --max-windows 1500000