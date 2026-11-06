#!/bin/bash
# step2_sweep.sh — STEP 2 (+ early stopping). 5 seeds x 2 models, identical
# protocol, early stopping patience 10, max 48 epochs. Resume-safe.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t2/step2
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_MAX_EPOCHS=96
export T2_PATIENCE=15
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
nice -n 19 ionice -c3 "$PY" scripts/t2_step2_sweep.py \
  >> runs/t2/step2/sweep.log 2>&1
echo "step2 rc=$?" >> runs/t2/step2/sweep.log