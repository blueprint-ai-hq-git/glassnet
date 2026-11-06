#!/bin/bash
# SMOKE for the K21n bypass (concept layer also reads the raw patches). Tiny (200 images, 3
# epochs, CPU, ~30 s) but it exercises EVERY path the change touches, and the one that matters
# most is g3: the manual explain() arithmetic must still match the forward pass exactly, now with
# the raw-patch concatenation present in BOTH. A mismatch here means the explanation would be
# describing a different computation than the model ran.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 32 --epochs 3 --n 200 --device cpu --lam-concept 0.6 --seed 0 \
  --concept-first 1 --concept-direct --out seetalk_k21n_smoke