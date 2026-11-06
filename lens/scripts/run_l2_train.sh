#!/bin/bash
# run_l2_train.sh — does the model train on the GPU, with the same numbers?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/tests/test_l2_train.py 2>&1 \
  | sed -n '/L2 TRAIN GATE/,$p' | tail -24