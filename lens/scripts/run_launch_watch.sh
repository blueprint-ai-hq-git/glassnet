#!/bin/bash
# Launcher for the watch loop (must be detached: the watch has to outlive this chat session).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/pipeline_watch.sh > /dev/null 2>&1 &
sleep 6
W=$(pgrep -f "[p]ipeline_watch.sh" | head -1)
echo "watch pid=${W:-NONE}"
ps -o pid,ppid,sid,cmd -p "${W:-1}" 2>/dev/null | tail -1
echo "--- newest heartbeats:"; tail -3 /tmp/pipeline_heartbeat.log 2>/dev/null
echo "--- supervisor:"; pgrep -af "[r]un_morning_supervisor" | head -1
echo "--- trainer:"; pgrep -af "[t]rain_chat.py" | head -1 | cut -c1-120
echo "--- newest checkpoint:"; ls -t lens/runs/chat/*_ep*.npz 2>/dev/null | head -1 | xargs basename