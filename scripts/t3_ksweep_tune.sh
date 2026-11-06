#!/bin/bash
# t3_ksweep_tune.sh — step 2 phase 1: TUNE lr for every K in {8,16,24,32},
# BOTH models, identical budget (lr in {0.3,0.5,1.0} x 1 seed), validation only.
#
# The frozen JOINT SELECTION RULE (SELECTION-RULE.md) is then applied by
# t3_ksweep_pick.py: lowest validation perplexity, tie-break on accuracy
# (within 1.0% relative perplexity), then lower lr.
#
# Matched parameters: the normal's hidden size == the readable's K, so both
# models have identical parameter counts at every K.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_MODELS=readable,normal
export T3_SEEDS=0
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
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