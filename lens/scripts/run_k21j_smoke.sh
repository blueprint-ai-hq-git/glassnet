#!/bin/bash
# SMOKE for the K21j path: tiny (200 images, 1 epoch, CPU) but it exercises EVERYTHING the real
# run does -- concept-first phase, the freeze, the early save, the gates, the CPU-mode g3 path.
# Measured reason: three K21j attempts died in the harness AFTER training finished, each losing
# the whole run. This costs 30 seconds and proves the path before another 6-minute GPU run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 32 --epochs 3 --n 200 --device cpu --lam-concept 0.6 --seed 0 \
  --concept-first 1 --out seetalk_k21j_smoke