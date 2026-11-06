#!/bin/bash
# Plain status script (no inline python, no pipes-with-loops: those trip the approval gate).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "time: $(date '+%F %H:%M:%S')"
echo "--- GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "--- jobs:"
pgrep -af "[t]rain_chat.py" | head -2
pgrep -af "[r]un_small_big_queue" | head -1
echo "--- queue log tail:"
tail -12 /tmp/small_big.log
echo "--- results file:"
tail -20 lens/runs/SMALL-BIG.txt 2>/dev/null
echo "--- newest checkpoints of the queue:"
ls -t lens/runs/chat/sb_*.npz 2>/dev/null | head -3