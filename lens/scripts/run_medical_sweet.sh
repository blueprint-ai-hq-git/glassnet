#!/bin/bash
# MEDICAL SWEET SPOT: seed-test lam_concept 0.05 and 0.10.
#
# WHAT IS NOW MEASURED (3 seeds each):
#   lam=0.00  acc 65.39 +- 1.33   concepts 13-16/35   accurate and STABLE, poorly readable
#   lam=0.15  acc 62.02 +- 4.51   concepts 22-30/35   readable but NOISY and ~3 pts lower
# Single-seed runs suggested 0.05 (67.35%) and 0.10 (67.33%) beat both. Single seeds have lied
# three times today, so both are now run on 3 seeds before any claim.
#
# GATES (pre-registered):
#   g1  the winner's mean accuracy must beat lam=0.00's 65.39 by MORE than the pooled std
#   g2  its variance must not be worse than lam=0.15's 4.51 (a readable model that swings wildly is
#       not shippable, so stability counts as a real requirement)
#   g3  its concept audit must be >= 20/35 (readability must actually be retained, or the point of
#       the concept layer is lost)
#   g4  if nothing satisfies all three, the honest answer is that no weight dominates and the
#       choice is a stated trade-off — NOT that one "won"
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

for LC in 0.05 0.10; do
  for SEED in 0 1 2; do
    echo ""
    echo "=== lam=$LC seed=$SEED ==="
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
      --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
      --lr 0.3 --lr-decay 0.5 --lam-concept $LC --seed $SEED --device cpu --color \
      --colour-concepts --adaptive --max-train 20000 --out "msweet_lc${LC}_s${SEED}" 2>&1 \
      | grep -E "final:|Error|Traceback" | tail -2
  done
done
echo ""
echo "=== SWEET SPOT SWEEP DONE ==="
