#!/bin/bash
# t3_lrgrid.sh — reviewer item (2): widen the lr grid for BOTH models.
#
#   lr in {0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}, K=64, 3 seeds, BOTH models,
#   validation only. Identical budget for every lr and both models.
#
# Then the best-vs-best comparison is recomputed from whichever lr each model
# selects on validation, and re-run at 5 seeds if either optimum moved off
# lr=0.3 (readable) / lr=0.1 (normal).
#
# Resume-safe: the runner skips (model, K, seed) triples already in its JSON,
# so re-running this script after an interruption continues rather than
# restarting. Item (3) is one comparison group -> one backend (GPU).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=64
export T3_MODELS=readable,normal
export T3_SEEDS=0,1,2
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/lrgrid.log

echo "=== lr grid (K=64, 7 lr x 2 models x 3 seeds)  $(date '+%F %T') ===" >> "$LOG"
for LR in 0.03 0.05 0.1 0.2 0.3 0.5 1.0; do
  echo "--- lr=$LR  $(date '+%T')" >> "$LOG"
  T3_LR=$LR T3_TAG="K64_lr${LR}_grid" \
    nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
  echo "    lr=$LR rc=$?" >> "$LOG"
done
echo "=== lr grid done  $(date '+%F %T') ===" >> "$LOG"