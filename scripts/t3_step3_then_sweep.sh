#!/bin/bash
# t3_step3_then_sweep.sh — re-run the fixed bypass check, then go straight into
# step 2 phase 1 (lr tuning for K in {8,16,24,32}, both models, equal budget).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"

echo "=== step 3 bypass (fixed)  $(date '+%F %T') ===" >> runs/t3/bypass.log
BYP_BACKEND=cupy BYP_K=64 BYP_LR=0.3 BYP_SEED=0 BYP_EPOCHS=96 \
BYP_MIN_EPOCH=30 BYP_PATIENCE=15 \
  nice -n 19 ionice -c3 "$PY" scripts/t3_bypass.py >> runs/t3/bypass.log 2>&1
echo "bypass rc=$?" >> runs/t3/bypass.log

# ---- step 2 phase 1 ----
export T3_BACKEND=cupy
export T3_MODELS=readable,normal
export T3_SEEDS=0
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
LOG=runs/t3/ksweep.log
echo "=== K sweep phase 1 (tuning)  $(date '+%F %T') ===" >> "$LOG"
for K in 8 16 24 32; do
  for LR in 0.3 0.5 1.0; do
    echo "--- K=$K lr=$LR  $(date '+%T')" >> "$LOG"
    T3_K=$K T3_LR=$LR T3_TAG="K${K}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    echo "    K=$K lr=$LR rc=$?" >> "$LOG"
  done
done
echo "=== phase 1 done  $(date '+%F %T') ===" >> "$LOG"