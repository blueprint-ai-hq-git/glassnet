#!/bin/bash
# Keeps the GPU fed: runs every line of GPU-QUEUE.txt in order, then writes a note that it drained.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while read -r line; do
  [ -z "$line" ] && continue
  echo "=== STARTING: $line  $(date '+%F %H:%M')" >> lens/runs/GPU-QUEUE-RUNS.txt
  /home/yahya/.local/bin/glassnet_chain.sh "$line" queueitem
  echo "=== DONE: $line  $(date '+%H:%M')  exit=$?" >> lens/runs/GPU-QUEUE-RUNS.txt
done < lens/runs/GPU-QUEUE.txt
echo "QUEUE DRAINED $(date '+%F %H:%M')" >> lens/runs/GPU-QUEUE-RUNS.txt
