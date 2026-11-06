#!/bin/bash
# T8b (code line): ONE change from the T8 run — epochs 32 -> 64 (the 32-epoch run got
# strict 27/35 and was still rising). Same corpus (clean v32000), d192, W16, lr 0.4 decay 0.5,
# lam 0.3, seed 0, gpu.
# BAR (pre-registered in LOG): strict label-honesty >= 35/35 (100%) AND syntax at n=150 >= 31
# (t8's own measured score — the beat-bar is the strict audit; syntax must not regress).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 64 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --out t8b_code_ep64 \
  --max-windows 1200000