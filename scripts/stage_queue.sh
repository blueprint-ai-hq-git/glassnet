#!/bin/bash
# stage_queue.sh — self-healing stage queue. Every stage is idempotent
# (resumes from checkpoints / skips finished configs), because external
# SIGKILLs hit tracked processes every ~15-25 min. Re-running this script
# continues from wherever it stopped; nothing restarts from scratch.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
export OPENBLAS_NUM_THREADS=6

# Stage 1: T2-004 probe (concat window, 192 epochs, seed 0). Pre-registered
# bar: val >= 20.0 AND audit >= 8/24. Resumes from the newest _ep*.npz.
if [ ! -f runs/t2/iter004/results.json ]; then
  T2_ITER=iter004 T2_EPOCHS=192 T2_SEEDS=0 T2_SAVE_EVERY=16 \
    nice -n 19 ionice -c3 $PY -u scripts/run_t2_iter003.py \
    >> runs/t2/iter004_run.log 2>&1
  echo "stage1 rc=$? $(date '+%H:%M:%S')" >> runs/queue.log
  touch runs/stage1.done
fi

# Stage 2: rung-3 FAIR sweep — tuned normal, VALIDATION only, test never
# read. Pre-registered verdict rule in tune_normal_banking77.py.
if [ ! -f runs/rung3_normal_tune/FINAL.txt ]; then
  nice -n 19 ionice -c3 $PY -u scripts/tune_normal_banking77.py \
    >> runs/rung3_normal_tune.log 2>&1
  echo "stage2 rc=$? $(date '+%H:%M:%S')" >> runs/queue.log
fi
echo "=== queue done $(date '+%F %T') ===" >> runs/queue.log