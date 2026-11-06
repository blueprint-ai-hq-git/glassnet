#!/bin/bash
# run_after_threshold.sh — wait for the running threshold bisect, then run the chain.
#
# One GPU, so everything is sequential. This waits for the threshold job's PID to exit, then runs
# the scaling test, the distillation study, and the benchmark summary.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the threshold bisect (pid $1) to finish..."
while kill -0 "$1" 2>/dev/null; do
  sleep 15
done
echo "threshold bisect finished; starting the chain"

bash lens/scripts/run_chain.sh 2>&1 | tail -70
