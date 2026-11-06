#!/bin/bash
# K21h (v2): the SAME 19-set / lam 0.6 / d128 / ep14 config as K21d, at seeds 1 and 2.
# v1 died instantly on an unrecognised --seed (the k22 trainer lacked the arg; now added,
# model-init only, datum stays seed 0 so the strict audit measures identical windows).
# WHY (measured): the two boundary concepts (com_low_w, left_right_symmetric_rel) cleared
# the strict margin at a different seed once already (K21b: symmetric_rel 0.126 honest), so
# the remaining shortfall is seed-dependent optimisation, not definition. Owner's doctrine:
# run the options, pick the best on evidence — and report the seed honestly, with the other
# seeds' results recorded beside it.
# BAR: strict 19/19 AND gates 4/4 in either seed -> that seed ships. Otherwise image stays
# 17/19 with the two named (already documented in BUGS.md).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
for S in 1 2; do
  echo "=== K21h seed $S ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
    --d 128 --epochs 14 --n 60000 --device gpu --lam-concept 0.6 --seed "$S" \
    --out "seetalk_k21h_s${S}"
done