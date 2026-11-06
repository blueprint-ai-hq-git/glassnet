#!/bin/bash
# SMOKE for the K21o mirror block (|I - mirror(I)| as a concept-layer input). Tiny (200 images, CPU,
# ~30 s) but it runs EVERY path the change touches; the one that matters is g3, which checks that
# the hand-written explain() arithmetic still matches the forward pass now that a THIRD input block
# exists. A mismatch here means the explanation would describe a different computation.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 32 --epochs 3 --n 200 --device cpu --lam-concept 0.6 --seed 0 \
  --concept-first 1 --concept-mirror --out seetalk_k21o_smoke