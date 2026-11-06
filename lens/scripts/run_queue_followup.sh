#!/bin/bash
# run_queue_followup.sh — chain the follow-ups after the code run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the code run (pid $1)..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "code run finished; starting the chat follow-ups"

bash lens/scripts/run_followup.sh 2>&1 | tail -60
