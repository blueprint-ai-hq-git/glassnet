#!/bin/bash
# T8f (code, push to 100%): ONE change from T8e — drop ONLY the last weak (mentions_count,
# margin 0.088) -> 24-set, lam stays 0.6 (the lever that gained 23->24). The drop-argument
# broke before ONLY when dropping 2 at once; here the set shrinks by one and lam stays high.
# BAR (pre-registered, binary): strict 24/24 (100%) AND val >= 38 (T8e's 38.33 holds).
# If a NEW weak appears it is named and the loop continues (this is the push, not a ceiling).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.6 --seed 0 --device gpu --out t8f_code24 \
  --max-windows 1200000