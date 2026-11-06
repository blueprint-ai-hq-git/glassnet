#!/bin/bash
# Detached launcher for the small->big queue (survives this chat session).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_small_big_queue.sh > /dev/null 2>&1 &
sleep 20
echo "queue pid=$(pgrep -f '[r]un_small_big_queue' | head -1)"
echo "trainer pid=$(pgrep -f '[t]rain_chat.py' | head -1)"
echo "--- first lines:"; tail -5 /tmp/small_big.log 2>/dev/null
echo "--- gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader