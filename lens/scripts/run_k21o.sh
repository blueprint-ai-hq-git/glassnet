#!/bin/bash
# K21o — THE NONLINEAR-BASIS TEST. ONE change from the best measured ship (K21j, concept-first,
# 18/19): the concept layer receives the |I - mirror(I)| map as an additional input block,
# available to EVERY concept neuron equally (each learns its own weights on it).
# WHY, from measurement rather than intuition:
#   * the single failing concept is left_right_symmetric_rel = mean|I - mirror(I)| / mean(I) --
#     a PAIRWISE, NON-LOCAL quantity; every other concept in the set is a local aggregate
#     (means of halves/quadrants, thresholds, edge magnitudes, centre-of-mass).
#   * its margin stayed pinned at ~0.05 across four architecture changes: d128/d160/d224 capacity
#     and direct raw-pixel access. A linear map over local features cannot form |I - mirror(I)|
#     at ANY width, which is exactly why those levers could not move it.
# DISCLOSED PLAINLY: this gives the model a feature block that makes a pairwise formula
# representable. It is raw material, not the formula's answer -- the neuron must still combine it
# with brightness itself (the formula is a RATIO), and the datum is held-out val windows, never
# training windows. The other 18 concepts get the same block and may ignore it.
# BAR (pre-registered): strict 19/19 AND gates 4/4. Anything less: image ships at 18/19 with all
# ten falsified levers named.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 \
  --concept-first 6 --concept-mirror --out seetalk_k21o_mirror