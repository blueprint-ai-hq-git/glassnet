#!/bin/bash
# MEDICAL: push the concept weight BELOW 0.15 — the trend was monotonic and I stopped too early.
#
# WHAT IS KNOWN (measured, replicated across 6 arms):
#   lam_concept 0.15 -> 66.03%   (best of the sweep)
#   lam_concept 0.30 -> 64.92%
#   lam_concept 0.50 -> 64.82%
# The trend is monotonic downward in weight, so 0.15 may NOT be the optimum — it was simply the
# lowest value tested. A monotonic trend that has not turned around means the optimum is unexplored.
#
# THIS RUN tests 0.00 (no concept loss at all — the control), 0.05, 0.10, and re-tests 0.15 as the
# anchor. d=64, everything else identical to the winning medical recipe.
#
# PRE-REGISTERED GATES (fixed BEFORE running):
#   n1  some weight < 0.15 beats 0.15's 66.03% by MORE than the seed noise (measured std 4.51 at
#       d64, so the bar is +4.5 pts, not +0.5 — a 1-point win is inside the noise and will NOT count)
#   n2  the winner must still beat its own plain control by > 10 pts
#   n3  weight 0.00 (no concept loss) is the CONTROL: if it matches the best weighted arm, the
#       concept loss is not what is helping and that must be reported
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for LC in 0.00 0.05 0.10 0.15; do
  echo ""
  echo "=== lam_concept=$LC ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
    --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
    --lr 0.3 --lr-decay 0.5 --lam-concept $LC --device cpu --color --colour-concepts \
    --adaptive --max-train 20000 --out "mlow_lc${LC}" 2>&1 \
    | grep -E "readable|plain|final|concepts|Error|Traceback" | tail -6
done
echo ""
echo "=== LOW-WEIGHT SWEEP DONE ==="
