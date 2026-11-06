#!/bin/bash
# GENERALIZATION ACROSS DATA SIZES — does the readable advantage hold at different training sizes?
# Only --max-train varies: 5000, 20000, 80000. Same dataset (Fashion-MNIST), same recipe.
# CPU only, runs alongside the GPU code-capacity experiment.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for N in 5000 20000 80000; do
  echo ""
  echo "=== SIZE TEST: max_train=$N ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
    --task vqa3 --data fashion --d 64 --patch 7 --epochs 8 --batch 256 \
    --lr 0.3 --lr-decay 0.5 --lam-concept 0.15 --device cpu --adaptive \
    --max-train $N --out "gensize_${N}" 2>&1 \
    | grep -E "readable|plain|final|concepts|Error|Traceback" | tail -6
done
echo ""
echo "=== SIZE TEST DONE ==="
