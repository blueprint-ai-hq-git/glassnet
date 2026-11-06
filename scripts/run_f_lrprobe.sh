#!/bin/bash
# run_f_lrprobe.sh — find a learning rate at which a student actually TRAINS on generated data.
# Same architecture, same corpus, same epochs; only lr changes. Whichever is used will be applied
# identically to every arm, so the comparison stays fair.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export F_EPOCHS=8
export F_PATIENCE=10
export F_TAG=lrprobe
for lr in 0.5 0.1 0.05 0.02; do
  echo "=== lr=$lr ==="
  F_KIND=normal F_RULE=all F_SEED=0 F_LR="$lr" F_LC=0.0 \
    nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/f_train.py 2>&1 \
    | grep -E "^  ep|STUDENT" | tail -4
done