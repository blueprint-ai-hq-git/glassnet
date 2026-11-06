#!/bin/bash
# K21m — the capacity step after K21l: d160 -> d224 (ONE change from the K21l run).
# MEASURED TREND (honest, two points): the one weak concept's margin under concept-first is
#      d128 (K21j):  own 0.794 vs rival 0.765 -> margin 0.029
#      d160 (K21l):  own 0.803 vs rival 0.749 -> margin 0.054
# Both runs: 0 mislabelled, AUROC all >= 0.8, 18/19. The margin moved +0.025 for +32 width.
# To clear the 0.10 bar the projected width is ~d224; this run measures that point. Extrapolation
# on two points is weak but cheap to test -- one run, one change, and the bar stays the same.
# BAR (pre-registered): strict 19/19 AND gates 4/4. Anything less: image ships at 18/19 with the
# honest distinction (0 mislabelled, the margin shortfall named with its number).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 224 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
  --concept-first 6 --out seetalk_k21m_d224