#!/bin/bash
# K21n — THE INFORMATION-BOTTLENECK TEST. ONE change from the best measured ship (K21j,
# concept-first, 18/19): the concept layer ALSO reads the raw patches (pixels in [0,1]) alongside
# the shared projection.
# WHY, stated before the run: the single shortfall is left_right_symmetric_rel, whose margin
# saturated at ~0.05 across d128/d160/d224 (own ~0.80, rival ~0.75). Capacity is exhausted, so the
# remaining hypothesis is that its signal does not survive the shared projection -- the neuron
# gets 128 numbers per patch and has to recover a fine left/right mass asymmetry from them. With
# the bypass it can read pixel structure directly; the neuron's explanation stays a plain weighted
# sum (projection features + raw pixels), so readability is preserved.
# VERIFIED BEFORE THE RUN: g3 (explain residual, the gate that checks the hand-written arithmetic
# still matches the forward) reads 0.00e+00 with the bypass active, in run_k21n_smoke.sh.
# BAR (pre-registered): strict 19/19 AND gates 4/4. Anything less: image ships at 18/19 with all
# nine falsified levers named.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
  --concept-first 6 --concept-direct --out seetalk_k21n_direct