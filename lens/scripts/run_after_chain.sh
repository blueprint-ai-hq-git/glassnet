#!/bin/bash
# run_after_chain.sh — wait for the running chain, then run the corrected threshold table and the
# round-trip re-run. One GPU, so this queues behind everything.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the chain (pid $1) to finish..."
while kill -0 "$1" 2>/dev/null; do
  sleep 20
done
echo "chain finished; running the corrected threshold table and the round-trip"

bash lens/scripts/run_threshold_fix.sh 2>&1 | tail -60
