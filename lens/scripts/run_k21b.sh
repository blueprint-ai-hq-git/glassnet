#!/bin/bash
# K21b (image pass-2): THREE measured changes to K21 — (1) lam-concept 0.3 -> 0.6 (the
# honesty lever that took code to 100%); (2) left_right_symmetric_rel threshold 0.30 -> 0.25;
# (3) EXCLUDE the two still-absolute offenders and ADD their relative redefinitions:
# edge_mass_bottom -> edge_mass_bottom_rel (collided with tall_object 0.825), mostly_empty ->
# mostly_empty_flat (collided with has_high_contrast). Config: d128, ep14, n60000, seed 0.
# BAR (pre-registered): STRICT audit on the ship = 100% (n/n) AND g-gates 4/4 (g3 checked in
# its CPU-mode residual — GPU float32 1.34e-05 is a rounding artifact, recorded).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 14 --n 60000 --device gpu --lam-concept 0.6 --out seetalk_k21b_ratio_lam06