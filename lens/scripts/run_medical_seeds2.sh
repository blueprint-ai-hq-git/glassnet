#!/bin/bash
# MEDICAL 3-SEED REPLICATION — the sweep winner (d96, lam_concept 0.15) vs the previous best
# (d64, lam_concept 0.15). A +0.71 point gap on ONE seed is not evidence; 3 seeds with a std is.
#
# PRE-REGISTERED (fixed BEFORE running):
#   s1  d96's mean must exceed d64's mean by more than the pooled std (a real gap, not noise)
#   s2  if the gap is inside the noise, declare "no difference found" and keep d64 (smaller, and
#       already proven) -- do NOT pick the larger model because it won once.
#   s3  every seed's readable-vs-plain advantage must stay > 10 pts (the readable claim must hold
#       on every seed, not just on average)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for cfg in "d64_lc0.15" "d96_lc0.15"; do
  D=$(echo "$cfg" | sed 's/d\([0-9]*\)_.*/\1/')
  for SEED in 0 1 2; do
    echo ""
    echo "=== $cfg seed $SEED ==="
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
      --task vqa3 --data pathmnist --d "$D" --patch 7 --epochs 8 --batch 256 \
      --lr 0.3 --lr-decay 0.5 --lam-concept 0.15 --seed "$SEED" --device cpu --color \
      --colour-concepts --adaptive --max-train 20000 --out "mseed_${cfg}_s${SEED}" 2>&1 \
      | grep -E "final:|TEST|Error|Traceback" | tail -4
  done
done
echo ""
echo "=== SEEDS DONE ==="
