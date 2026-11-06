#!/bin/bash
# RETRAIN with the FIXED concept-BCE gradient (2026-10-10).
# The old backward was wrong by factor pos_weight for every concept with support != 0.5; the medical
# readable arm (lam=0.15) trained under that distortion, so its accuracy and the +24.62-vs-plain
# claim MUST be re-measured before they can stand. Same recipe as run_medical_flat.sh, same gates.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for LC in 0.00 0.15; do
  for SEED in 0 1 2; do
    echo ""
    echo "=== FIXED-GRADIENT lam=$LC seed=$SEED ==="
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
      --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
      --lr 0.3 --lr-decay 0.5 --lam-concept $LC --seed $SEED --device cpu --color \
      --colour-concepts --adaptive --max-train 20000 --out "mfix_lc${LC}_s${SEED}" 2>&1 \
      | grep -E "final:|concepts [0-9]|Error|Traceback" | tail -3
  done
done
echo ""
echo "=== FIXED-GRADIENT MEDICAL RETRAIN DONE ==="