#!/bin/bash
# run_g.sh — gradient check first (project rule), then the Target G loop.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
LOG=runs/t3/g_loop.log
mkdir -p runs/t3

echo "=== GRADIENT CHECK $(date '+%T') ===" | tee "$LOG"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g_gradcheck.py 2>&1 | tail -12 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== SMOKE (1 seed, 8 epochs) $(date '+%T') ===" | tee -a "$LOG"
G_SEEDS=0 G_EPOCHS=8 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g_loop.py 2>&1 \
  | tail -20 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== FULL (5 seeds, 60 epochs) $(date '+%T') ===" | tee -a "$LOG"
G_SEEDS=0,1,2,3,4 G_EPOCHS=60 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g_loop.py 2>&1 \
  | tee -a "$LOG" | tail -28