#!/bin/bash
# t3_stageB.sh — item (3) confirmation: each model at its OWN best config,
# chosen on VALIDATION by the equal-budget tuning sweep (t3_tune.sh), 3 seeds.
#
#   readable -> K=64, lr=0.3   (its val-selected best: 30.93% / ppl 58.74)
#   normal   -> K=64, lr=0.1   (its val-selected best: 30.72% / ppl 63.70)
#
# Both got the identical tuning budget (2 lr values x 1 seed). This is
# best-vs-best, which is the fair headline comparison — Stage A compared both
# at lr=0.1, which is the readable's WORSE learning rate.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=64
export T3_SEEDS=0,1,2
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/stageB.log

echo "=== stage B (best-vs-best, K=64, 3 seeds)  $(date '+%F %T') ===" >> "$LOG"
T3_MODELS=readable T3_LR=0.3 T3_TAG=K64_best_readable \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "readable rc=$?" >> "$LOG"
T3_MODELS=normal T3_LR=0.1 T3_TAG=K64_best_normal \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "normal rc=$?" >> "$LOG"
echo "=== stage B done  $(date '+%F %T') ===" >> "$LOG"