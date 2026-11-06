#!/bin/bash
# THE REAL-CORPUS DISTILLATION RUN (owner's question: can our best model make a smaller one
# smarter?). Teacher = the shipped talk line t11_talk41_ep192 (d256, 30.9% plain / 80.6%
# assistant-masked on its own words). Students d128, assistant-only (the teacher's own
# objective), 400k train windows, 8 epochs, seeds 0/1/2, soft-KL 0.5 for arm A only.
# Pre-registered predictions are in run_distill_real.py's header (P1-P4) and were written
# BEFORE this run; the bar is A > B by > 2 pts with every seed positive.
# Waits for the GPU: K21h (image seeds 1/2) runs first -- one GPU, everything sequential.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CD_VOCAB=12000
# wait loop removed (pgrep -f phantom match): K21h has finished, the GPU is free
echo "GPU free; starting the real-corpus distillation"
export DR_DEVICE=gpu DR_WINDOWS=400000 DR_EPOCHS=8 DR_SEEDS=0,1,2 DR_LAM_SOFT=0.5
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/run_distill_real.py