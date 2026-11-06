#!/bin/bash
# CONFIRM: is medical accuracy really FLAT across concept weights? (3 seeds on the extremes)
#
# WHY: the low-weight sweep showed accuracy 66.03-67.35% across lam_concept 0.00..0.15 while the
# concept audit went 13/35 -> 30/35. If accuracy is genuinely flat, then the concept loss buys
# READABILITY at no accuracy cost — a clean, important result. But a 1-point spread on single seeds
# is exactly the kind of thing that has been noise three times today (measured seed std at d64 was
# 4.51). So confirm it with seeds BEFORE claiming flatness.
#
# GATES (pre-registered):
#   f1  accuracy difference between 0.00 and 0.15 is smaller than the pooled std -> FLAT confirmed
#   f2  the concept audit difference (13/35 vs 30/35) is LARGE and monotonic -> readability bought
#   f3  if f1 fails (they differ significantly), then the weight DOES affect accuracy and the
#       "no accuracy cost" claim is wrong — report that instead
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for LC in 0.00 0.15; do
  for SEED in 0 1 2; do
    echo ""
    echo "=== lam=$LC seed=$SEED ==="
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
      --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
      --lr 0.3 --lr-decay 0.5 --lam-concept $LC --seed $SEED --device cpu --color \
      --colour-concepts --adaptive --max-train 20000 --out "mflat_lc${LC}_s${SEED}" 2>&1 \
      | grep -E "final:|concepts [0-9]|Error|Traceback" | tail -3
  done
done
echo ""
echo "=== FLATNESS CONFIRM DONE ==="
