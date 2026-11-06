#!/bin/bash
# TRACK 3 (code line): epochs 16 -> 32, ONE change from the best code model (cclean32k_ep16).
# Same corpus (clean, v32000), d192, W16, lr 0.4 decay 0.5, lam 0.3, seed 0.
# GATE (pre-registered): verdict_code_clean.measure at n=150 (own vocab) MUST beat 30/150 syntax.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --out t8_code_ep32 --max-windows 1200000 \
  2>&1 | tee lens/runs/code/t8_code_ep32.txt