#!/bin/bash
# K21l — CONCEPT-FIRST (proven, 18/19) at d=160. ONE change from the best measured ship (K21j):
# the per-patch embedding width 128 -> 160.
# WHY THIS AND NOT A LONGER PHASE, stated before the run: the concept-phase loss in K21j went
# 0.1152 -> 0.0640 -> 0.0562 -> 0.0520 -> 0.0493 -> 0.0477, i.e. it PLATEAUS by epoch 6 while the
# phase still had 0 epochs left. A limit that the optimisation has already reached is not a limit
# of time, it is a limit of representation -- so the lever to test is capacity, not epochs.
# d=160 was falsified ONCE BEFORE, but under the OLD objective (answer loss on from epoch 1).
# The combination concept-first + d160 has never been measured.
# BAR (pre-registered): strict 19/19 AND gates 4/4. Anything less is named, and image ships at
# 18/19 with the honest distinction: 0 mislabelled (every neuron's top correlation IS its own
# formula) and one margin shortfall (left_right_symmetric_rel, 0.794 vs 0.765 = 0.029 < 0.10).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 160 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
  --concept-first 6 --out seetalk_k21l_d160