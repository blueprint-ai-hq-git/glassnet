#!/bin/bash
# run_g_loop.sh — Target G: the verifiable-task self-improvement loop.
# Teacher trains on a fixed 70 of 100 problems, answers all 100 twenty times each, the verifier
# keeps only correct answers, three student arms are compared on the 30 held-out problems.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
LOG=runs/t3/g_loop.log
mkdir -p runs/t3

# smoke first: 1 seed, few epochs -- prove the pipeline runs end to end before the 5-seed run
echo "=== SMOKE (1 seed, 8 epochs) $(date '+%T') ===" | tee "$LOG"
G_SEEDS=0 G_EPOCHS=8 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g_loop.py 2>&1 \
  | tail -22 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== FULL (5 seeds, 60 epochs) $(date '+%T') ===" | tee -a "$LOG"
G_SEEDS=0,1,2,3,4 G_EPOCHS=60 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g_loop.py 2>&1 \
  | tee -a "$LOG" | tail -30