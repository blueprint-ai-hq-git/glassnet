#!/bin/bash
# K21j — THE CONCEPT-FIRST EXPERIMENT (one structural change, not a re-tune).
# Why: every TUNING lever for the image boundary concepts has been falsified (pruning is the
# wrong tool here; seeds 1/2 identical; capacity d160 identical; epochs 20 identical). So the
# remaining lever is structural: stop making the concept neurons compete with the answer
# objective. For the first 6 epochs the answer loss is switched OFF (concept loss only, the
# neurons' whole job is to match their own formulas), then the front+concept layer is FROZEN
# and only the answer layer trains.
# Same 19-set, lam 0.6, d128, seed 0, datum untouched (the strict audit pins its own seed).
# BAR (pre-registered): strict 19/19 AND gates 4/4. If 19/19 is reached, image ships at 100%;
# if it fails, the failure is named like every other lever and the next structural option is
# the orthogonality (rivalry) penalty.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
  --concept-first 6 --out seetalk_k21j