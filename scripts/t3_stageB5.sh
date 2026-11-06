#!/bin/bash
# t3_stageB5.sh — extend Stage B from 3 to 5 seeds (adds seeds 3 and 4).
# Same configs: readable K=64/lr=0.3, normal K=64/lr=0.1. Resume-safe: the
# runner skips (model, K, seed) triples already in the JSON.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=64
export T3_SEEDS=3,4
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/stageB.log

echo "=== stage B -> 5 seeds (adding 3,4)  $(date '+%F %T') ===" >> "$LOG"
T3_MODELS=readable T3_LR=0.3 T3_TAG=K64_best_readable \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "readable rc=$?" >> "$LOG"
T3_MODELS=normal T3_LR=0.1 T3_TAG=K64_best_normal \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "normal rc=$?" >> "$LOG"
echo "=== 5-seed extension done  $(date '+%F %T') ===" >> "$LOG"