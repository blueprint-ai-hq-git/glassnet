#!/bin/bash
# K21d (image, FINAL): changes vs K21c — (1) revert left_right_symmetric_rel threshold to
# 0.30 (0.25 broke honesty in K21c: margin 0.078; 0.30 held 0.126 in K21b), (2) DROP
# is_thin_structure (margin declined 0.110 -> 0.042 -> -0.083 across K21/K21b/K21c while its
# rival has_high_contrast_rel rose; a fraction-window formula a structural rival keeps
# beating — measured fragile across three independent nets, dropped per never-forced).
# Set: 19 concepts = the 17 K21b-honest + com_high_w/com_low_w/has_high_contrast_rel - is_thin_structure.
# lam 0.6, d128, ep14, n60000, seed 0. BAR (pre-registered): STRICT 19/19 AND gates 4/4.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 160 --epochs 14 --n 60000 --device gpu --lam-concept 0.6 --out seetalk_k21g