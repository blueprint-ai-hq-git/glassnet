#!/bin/bash
# t3_stageC.sh — reviewer item (2) tail: both optima MOVED, so the best-vs-best
# is re-run at 5 seeds on the moved optima, plus the readable's
# perplexity-optimal lr for the sensitivity report.
#
#   readable lr=0.5  (pre-registered selection: max validation accuracy)
#   readable lr=1.0  (its perplexity-optimal -- reported as sensitivity only)
#   normal   lr=0.2  (its selection: max validation accuracy; moved from 0.1)
#
# Seeds 0-2 already exist from the grid; this adds seeds 3 and 4.
# Resume-safe: the runner skips triples already in the JSON.
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
LOG=runs/t3/lrgrid.log

echo "=== stage C: 5-seed extension on the moved optima  $(date '+%F %T') ===" >> "$LOG"
T3_LR=0.5 T3_MODELS=readable T3_TAG="K64_lr0.5_grid" \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "readable lr0.5 rc=$?" >> "$LOG"
T3_LR=1.0 T3_MODELS=readable T3_TAG="K64_lr1.0_grid" \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "readable lr1.0 rc=$?" >> "$LOG"
T3_LR=0.2 T3_MODELS=normal T3_TAG="K64_lr0.2_grid" \
  nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
echo "normal lr0.2 rc=$?" >> "$LOG"
echo "=== stage C done  $(date '+%F %T') ===" >> "$LOG"