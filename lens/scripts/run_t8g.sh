#!/bin/bash
# T8g (code, final): ONE change from T8f — lam-concept 0.6 -> 0.3 (the accuracy-optimal lam;
# T8f proved the 24-set at lam 0.6 = strict 24/24 but val 37.69 vs T8e's 38.33 at 0.3 on the
# 25-set). Same 24-set, ep32, d192, seed 0.
# BAR (pre-registered, binary): strict 24/24 AND val >= 38.33 (beat T8e).
# If it passes -> ships as the 100% code line at max val. If not -> T8f ships (100%, 37.69).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --out t8g_code24_lam03 \
  --max-windows 1200000