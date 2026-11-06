#!/bin/bash
# RETRAIN every remaining AFFECTED shipped artifact on the corrected concept-BCE gradient
# (2026-10-10). The old backward was wrong by factor pos_weight for support != 0.5:
#   SEE-ONLY ship (K23): 0/23 concepts at w==1, w up to 48.9
#   SEE-ONLY ladder (K13): 0/13, w up to 48.9
#   TALK ship (asonly_v12000): 0/41 at w==1, w up to 170
# Each uses exactly its original launcher (proven loops untouched), only the engine's gradient is
# now correct. Outputs are NEW names asonly_v12000_fixedgrad / lad_image_cls_fixedgrad /
# ktest_real_{arm}_s{seed}_fixedgrad — the old artifacts are KEPT.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== 1/3 SEE-ONLY ladder (K13) retrain with the corrected gradient ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
  --d 64 --epochs 24 --batch 256 --lr 0.3 --lr-decay 0.5 --lam-concept 0.3 \
  --device gpu --out lad_image_cls_fixedgrad 2>&1 \
  | grep -E "readable .*TEST|plain .*TEST|concepts [0-9]|Error|Traceback" | tail -4

echo "=== 2/3 SEE-ONLY ktest (K23, 3 arms x 3 seeds) retrain with the corrected gradient ==="
for arm in none real dummy; do
  for seed in 0 1 2; do
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
      --epochs 12 --device gpu --extra "$arm" --no-plain --seed "$seed" \
      --out "ktest_${arm}_s${seed}_fixedgrad" 2>&1 \
      | grep -E "EXTRA|TEST|Error|Traceback" | tail -2
  done
done

echo "=== 3/3 TALK (asonly v12000) retrain with the corrected gradient ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 16 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --assistant-only --out asonly_v12000_fixedgrad \
  --max-windows 1500000 2>&1 \
  | grep -E "^  ep |final|ai[0-9]|Error|Traceback" | tail -12

echo "=== ALL FIXED-GRADIENT RETRAINS DONE ==="