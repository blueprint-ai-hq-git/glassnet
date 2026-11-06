#!/bin/bash
# GENERALIZATION TEST: does the readable advantage carry to a SECOND domain?
#
# WHY THIS MATTERS: every strong result so far is on pathmnist. One dataset is not evidence that the
# readable architecture is better in general — it could be a lucky pairing with histopathology.
# The owner's rule is "test whether improvements carry", so run the SAME recipe on Fashion-MNIST,
# a completely different domain (grey 28x28 clothing photos vs colour tissue patches).
#
# HELD IDENTICAL to the winning medical recipe: d=64, patch 7, 8 epochs, lr 0.3 decay 0.5,
# lam_concept 0.15, adaptive, 20k train samples. Only the DATA changes. One change at a time.
#
# PRE-REGISTERED GATES (fixed BEFORE running):
#   z1  readable beats its own plain control by > 5 pts on Fashion-MNIST
#       (bar is lower than medical's +24.6 because Fashion-MNIST is a harder 10-class problem)
#   z2  the direction is the same as medical (readable > plain). If readable LOSES here, the
#       advantage does NOT generalize and that must be reported plainly, not buried.
#   z3  concept audit pass rate reported either way, so a win cannot hide broken concepts.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== GENERALIZATION: Fashion-MNIST, same recipe as med_cc (d=64, lam 0.15) ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data fashion --d 64 --patch 7 --epochs 8 --batch 256 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.15 --device cpu --adaptive \
  --max-train 20000 --out "gen_fashion_d64" 2>&1 \
  | grep -E "readable|plain|TEST|final|concepts|Error|Traceback" | tail -10

echo ""
echo "=== and the colour variant (fashion is grey; tests whether --color helps or hurts) ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data fashion --d 64 --patch 7 --epochs 8 --batch 256 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.15 --device cpu --color --colour-concepts \
  --adaptive --max-train 20000 --out "gen_fashion_colour" 2>&1 \
  | grep -E "readable|plain|TEST|final|concepts|Error|Traceback" | tail -10
echo "=== DONE ==="
