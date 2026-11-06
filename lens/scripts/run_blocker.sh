#!/bin/bash
# run_blocker.sh — the rebuilt engine against the perplexity blocker, on the GPU.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_EPOCHS="${BLOCKER_EPOCHS:-30}"
mkdir -p lens/runs
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/run_blocker.py 2>&1 | tee lens/runs/blocker_run.log