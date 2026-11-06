#!/bin/bash
# RE-VERIFY the E17 generalization claim (+6.74 on Fashion-MNIST) on the CORRECTED gradient.
# WHY: the concept-BCE backward was wrong by a factor of pos_weight for support != 0.5. The E17 arms
# trained with lam_concept 0.15 and real supports, so the +6.74 figure is SUSPECT until re-measured —
# exactly as the k1 K-transfer gate flipped from PASS to FAIL when re-run on the fixed engine.
# Same recipe, same gates, same seed. Only the engine's gradient is now correct.
#   z1 readable beats its plain control by > 5 pts
#   z2 direction is the same as medical (readable > plain); if it LOSES, report it plainly
#   z3 concept audit pass rate reported either way
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== FIXED-GRADIENT GENERALIZATION: Fashion-MNIST, same recipe as med_cc (d=64, lam 0.15) ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data fashion --d 64 --patch 7 --epochs 8 --batch 256 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.15 --device cpu --adaptive \
  --max-train 20000 --out "gen_fashion_d64_fixedgrad" 2>&1 \
  | grep -E "readable|plain|TEST|final|concepts|Error|Traceback" | tail -10
echo "=== FIXED-GRADIENT GENERALIZATION DONE ==="