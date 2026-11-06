#!/bin/bash
# run_f_control.sh — THE CONTROL: identical script, identical hyperparameters, REAL data.
# If this does not reach ~80 ppl / ~30% acc like the teacher, the training script is broken and
# no conclusion about generated data can be trusted.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export F_EPOCHS=12
export F_PATIENCE=12
export F_TAG=control
F_REAL_DATA=1 F_KIND=normal F_RULE=all F_SEED=0 F_LR=1.0 F_LC=0.0 \
  nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/f_train.py 2>&1 | tail -10