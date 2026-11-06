#!/bin/bash
# K21c (image pass-3): ONE class of change vs K21b — the last 3 weak concepts replaced with
# darkness-fair redefinitions: com_high -> com_high_w, com_low -> com_low_w (luminance-
# weighted centres of mass), has_high_contrast -> has_high_contrast_rel (std/mean).
# Everything else identical: lam 0.6, d128, ep14, n60000, seed 0.
# BAR (pre-registered, binary): STRICT audit 20/20 (100%) AND gates 4/4 (g3 in CPU-residual
# mode). If a NEW weak appears it is named and looped — this is the push to 100%.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 14 --n 60000 --device gpu --lam-concept 0.6 --out seetalk_k21c_lam06