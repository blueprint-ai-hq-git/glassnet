#!/bin/bash
# SMOKE for run_distill_real.py: tiny slice, CPU, 1 epoch, 1 seed. Catches shape/vocab/op
# errors in a minute instead of 20 minutes into a GPU run. Prints the per-step loss line so
# the soft-KL term is visibly exercised.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CD_VOCAB=12000
export DR_DEVICE=cpu DR_WINDOWS=3000 DR_EPOCHS=1 DR_SEEDS=0 DR_OUT=distill_real_smoke
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/run_distill_real.py