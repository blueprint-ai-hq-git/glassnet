#!/bin/bash
# run_queue_code.sh — wait for the combined chat run, then start the code model.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for best1 (pid $1)..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "best1 finished; starting the code model"

bash lens/scripts/run_code_real.sh 2>&1 | tail -75
