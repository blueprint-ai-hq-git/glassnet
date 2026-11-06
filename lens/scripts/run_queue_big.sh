#!/bin/bash
# run_queue_big.sh — wait for the distillation seeds, then start the bigger chat run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the distillation seeds (pid $1)..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "seeds finished; starting the bigger chat run"

bash lens/scripts/run_chat_big.sh 2>&1 | tail -95
