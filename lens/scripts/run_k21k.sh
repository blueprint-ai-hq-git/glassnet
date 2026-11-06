#!/bin/bash
# K21k — CONCEPT-FIRST (proven: 17/19 -> 18/19) PLUS THE RIVALRY PENALTY, aimed at the ONE
# concept still weak: left_right_symmetric_rel (0.798 vs top_left_mass 0.768, margin 0.030).
# The decorr term penalises the squared off-diagonal covariance of the concept activations --
# i.e. exactly the rivalry the strict audit measures -- and its gradient was numerically checked
# (decorr.py, analytic vs finite differences 1.25e-9) BEFORE being wired.
# ONE added term relative to K21j; the concept-first phase, the set, the datum and every other
# setting are identical, so a change in the verdict is attributable to the penalty.
# TWO lambda values are run (owner's doctrine: run the options, pick the best on evidence).
# BAR (pre-registered): strict 19/19 AND gates 4/4. Anything less is named and reported.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
for LAM in 0.3 1.0; do
  echo "=== K21k decorr lambda $LAM ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
    --d 128 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
    --concept-first 6 --decorr "$LAM" --out "seetalk_k21k_d${LAM}"
done