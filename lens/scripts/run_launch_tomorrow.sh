#!/bin/bash
# Morning launcher: puts everything back up from where last night stopped.
# Detached (setsid) so it survives this chat. Order: resume B1 -> corpus arms -> 192-epoch arms.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_tomorrow.sh > /tmp/tomorrow.log 2>&1 &
sleep 45
echo "chain pid=$(pgrep -f '[r]un_tomorrow' | head -1)"
echo "trainer: $(pgrep -af '[t]rain_chat.py' | head -1 | cut -c1-110)"
echo "--- log so far:"; tail -12 /tmp/tomorrow.log 2>/dev/null
echo "--- gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader