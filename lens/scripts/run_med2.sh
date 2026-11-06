#!/bin/bash
# MED-2 (medical line, pass 1): two measured changes from med_cc (the current ship whose
# strict audit measured honest 1/31):
#   (1) COLOUR_ONLY — the readable layer is the colour concepts alone (strict audit: 12/13
#       grayscale concepts mislabelled/weak; neurons latch RGB axes; histology signal IS colour)
#   (2) lam-concept 0.15 -> 0.6 (the honesty lever proven on code 24/24 and image 17/19-ish)
# Everything else identical to med_cc: task vqa3, pathmnist, d64, patch7, ep8, batch256,
# lr 0.3, lr-decay 0.5, seed 0, cpu, --color --colour-concepts.
# BAR (pre-registered, binary): STRICT audit on the new ship = n/n honest AND every image
# concept AUROC >= 0.8. If a new weak appears it is named and looped (never-forced drops
# come with dossiers, after the retrain re-roll is measured).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 COLOUR_ONLY=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --patch 7 --epochs 8 --batch 256 \
  --lr 0.3 --lr-decay 0.5 --lam-concept 0.6 --seed 0 --device cpu \
  --color --colour-concepts --out med_cc2