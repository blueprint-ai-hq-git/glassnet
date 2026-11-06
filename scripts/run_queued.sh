#!/bin/bash
# run_queued.sh — detached worker: runs T2-004 to completion, then the
# rung-3 fair-comparison sweep. setsid+nohup so no session/harness
# lifecycle can SIGKILL it (three long runs died to external kills today).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3

echo "=== queue start $(date '+%F %T') ===" >> runs/queue.log

# 1) T2-004 probe: concat window, 192 epochs, seed 0 (pre-registered)
if [ ! -f runs/t2/iter004/results.json ]; then
  echo "--- T2-004 start $(date '+%H:%M:%S')" >> runs/queue.log
  T2_ITER=iter004 T2_EPOCHS=192 T2_SEEDS=0 OPENBLAS_NUM_THREADS=6 \
    nice -n 19 ionice -c3 $PY -u scripts/run_t2_iter003.py \
    >> runs/t2/iter004_run.log 2>&1
  echo "--- T2-004 done rc=$? $(date '+%H:%M:%S')" >> runs/queue.log
fi

# 2) rung-3 fair sweep: tuned normal, VAL only, test untouched
echo "--- normal sweep start $(date '+%H:%M:%S')" >> runs/queue.log
OPENBLAS_NUM_THREADS=6 nice -n 19 ionice -c3 \
  $PY -u scripts/tune_normal_banking77.py >> runs/rung3_normal_tune.log 2>&1
echo "--- normal sweep done rc=$? $(date '+%H:%M:%S')" >> runs/queue.log
echo "=== queue end $(date '+%F %T') ===" >> runs/queue.log