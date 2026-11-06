#!/bin/bash
# Data-judge SMOKE: tiny slice, 5 epochs, CPU. Catches feature/label/engine errors in a minute.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CD_VOCAB=12000
export DJ_N_TRAIN=1500 DJ_N_VAL=800 DJ_EPOCHS=5
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/data_judge.py