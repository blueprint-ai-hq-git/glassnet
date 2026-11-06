#!/bin/bash
# run_queue_char.sh — wait for the optimisation grid, then run the full character-level model.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the optimisation grid (pid $1)..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "grid finished; starting the character-level run"

bash lens/scripts/run_chat_char_real.sh 2>&1 | tail -70
