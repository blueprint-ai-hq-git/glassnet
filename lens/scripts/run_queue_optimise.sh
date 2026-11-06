#!/bin/bash
# run_queue_optimise.sh — wait for the currently-queued big run, then start the grid.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the queued big run (pid $1)..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "big run finished; starting the optimisation grid"

bash lens/scripts/run_optimise.sh 2>&1 | tail -80
