#!/bin/bash
# MEDICAL SWEEP — all reasonable options, pick the best on held-out evidence.
#
# WHY: medical is the STRONGEST capability in this project (med_cc: readable 66.03% vs plain
# 41.41%, +24.62 pts, 30/35 concepts firing). It is also CHEAP: 2.67s/epoch on CPU, no GPU needed.
# So instead of guessing which config is best, run the plausible set and let held-out accuracy decide.
#
# WHAT VARIES (one axis per arm, everything else held at the med_cc recipe that already won):
#   width d        64 (current) | 96 | 128
#   concept weight 0.15 (current) | 0.30 | 0.50
#   seeds          0,1,2 on the best arm afterwards (never claim a win from one seed)
#
# PRE-REGISTERED GATES (fixed BEFORE running):
#   m1  some arm beats med_cc's 66.03% readable test accuracy
#   m2  the winner beats its OWN plain control by > 10 pts (the readable advantage must survive)
#   m3  the winner's advantage replicates on 3 seeds (mean +- std, not a single lucky run)
#   m4  if no arm beats 66.03%, report "no improvement found" — do NOT reword it as a success.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

RUN=1
for cfg in "d64_lc0.15" "d96_lc0.15" "d128_lc0.15" "d64_lc0.30" "d96_lc0.30" "d64_lc0.50"; do
  D=$(echo "$cfg" | sed 's/d\([0-9]*\)_.*/\1/')
  LC=$(echo "$cfg" | sed 's/.*lc//')
  echo ""
  echo "=== ARM $RUN: $cfg  (d=$D lam_concept=$LC) ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
    --task vqa3 --data pathmnist --d "$D" --patch 7 --epochs 8 --batch 256 \
    --lr 0.3 --lr-decay 0.5 --lam-concept "$LC" --device cpu --color --colour-concepts \
    --adaptive --max-train 20000 --out "msweep_$cfg" 2>&1 \
    | grep -E "readable|plain|TEST|final|Error|Traceback" | tail -6
  RUN=$((RUN+1))
done

echo ""
echo "=== SWEEP DONE — results in lens/runs/vl/msweep_*_log.json ==="
