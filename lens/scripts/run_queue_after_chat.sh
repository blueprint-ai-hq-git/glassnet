#!/bin/bash
# run_queue_after_chat.sh — run the distillation seeds after the chat run frees the GPU.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the chat training (pid $1) to finish..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "chat run finished; starting the distillation seeds"

bash lens/scripts/run_distill_seeds.sh 2>&1 | tail -40
