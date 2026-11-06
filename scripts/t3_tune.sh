#!/bin/bash
# t3_tune.sh — item (3) K sweep with EQUAL validation-only tuning.
#
# For every K in {24, 64, 128, 256} and BOTH models, the SAME tuning budget:
#     lr in {0.1, 0.3}  x  1 seed, selected on VALIDATION accuracy.
# Then the winning config per (K, model) is confirmed with 3 seeds.
#
# Equal budget for every K and both models is the point: it is the mistake that
# invalidated the rung-3 comparison (the readable was tuned, the normal was not).
#
# Item (3) is one comparison group, so everything runs on the GPU.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/tune.log

echo "=== K sweep (equal VAL-only tuning)  $(date '+%F %T') ===" >> "$LOG"
for K in 24 64 256; do
  for LR in 0.1 0.3; do
    echo "--- K=$K lr=$LR (tuning, 1 seed, val only)  $(date '+%T')" >> "$LOG"
    T3_K=$K T3_SEEDS=0 T3_MODELS=readable,normal T3_LR=$LR \
      T3_TAG="K${K}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    rc=$?" >> "$LOG"
  done
done
echo "=== tuning stage done  $(date '+%F %T') ===" >> "$LOG"