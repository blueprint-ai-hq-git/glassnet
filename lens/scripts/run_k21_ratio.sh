#!/bin/bash
# K21 (image see+talk, ratio-relative redesign): ONE change from the K22 big ship — concept
# set 22 -> 21 (EXCLUDE the 8 collinear offenders, ADD the 6 ratio-relative formulas:
# bottom_left_mass_rel, bottom_right_mass_rel, top_right_mass_rel, centre_column_rel,
# centre_row_rel, left_right_symmetric_rel). Config identical: d128, epochs 14, n 60000, seed 0.
# PRE-GATE note (recorded): 101 formula pairs > 0.2 — all top pairs are concepts already
# strict-honest (is_dark 0.971 / is_bright 0.928 / top_left_mass 0.943); the <=0.2 formula bar
# cannot hold on fashion data (brightness/mass/ink co-occur). Ship gate = STRICT AUDIT 21/21
# (own vs best-other margin > 0.10) + the trainer's g1-g4 + delta vs text-only control.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 14 --n 60000 --device gpu --out seetalk_k21_ratio