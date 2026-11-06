#!/bin/bash
# MED-3 (medical line, pass 2): ONE change from MED-2 (colour-only, lam 0.6) —
# the colour concept set pruned from 15 to the 4 measured as separable at
# max-pair |formula corr| = 0.565: red_exceeds_blue, red_vs_blue_top_bottom,
# red_channel_high, is_purple_dominant (see LOG: 12/105 pairs > 0.8, one pair 1.000).
# Everything else identical to med_cc2: vqa3, pathmnist, d64, patch7, ep8, batch256,
# lr 0.3, lr-decay 0.5, lam 0.6, seed 0, cpu, --color --colour-concepts.
# BAR (pre-registered): STRICT audit n/n (4/4) on the new ship, all AUROC >= 0.8,
# answer stays > plain + majority.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 COLOUR_ONLY=1
export COLOUR_SELECT="red_exceeds_blue,red_vs_blue_top_bottom,red_channel_high,is_purple_dominant"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.6 --seed 0 --device cpu \
  --color --colour-concepts --out med_cc3