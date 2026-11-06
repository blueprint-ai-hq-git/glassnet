#!/bin/bash
# RETRAIN the SHIPPING config (seetalk_k22_big) with the FIXED concept-BCE gradient (2026-10-10).
# The old backward was wrong by factor pos_weight for all 22 concepts (supports 0.02..0.97 -> w
# 0.03..48.7, ZERO of them w==1), so the shipped model's concept layer learned under a distorted
# gradient. Its val accuracy and the K22-winning delta MUST be re-measured before the shipped
# artifact can be trusted. Same recipe as train_seetalk_k22.py defaults (d=128, 14 ep, n=60000,
# lam_concept=0.3) so the numbers are directly comparable.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 14 --n 60000 --device gpu --out seetalk_k22_big_fixedgrad \
  2>&1 | tail -25
echo "=== K22 FIXED-GRADIENT RETRAIN DONE ==="