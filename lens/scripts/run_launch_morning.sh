#!/bin/bash
# Launch the morning pipeline DETACHED (own session, no controlling terminal), so it keeps running
# independently of the chat session and can be resumed after any shutdown.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_morning_supervisor.sh > /dev/null 2>&1 &
sleep 8
PID=$(pgrep -f "[r]un_morning_supervisor" | head -1)
echo "morning supervisor pid=${PID:-NONE}"
ps -o pid,ppid,sid,etime,cmd -p "${PID:-1}" 2>/dev/null | tail -1
echo "--- log:"; head -4 /tmp/morning.log 2>/dev/null
echo "--- trainer up?"; pgrep -f "[t]rain_chat.py .*t12_attn_d512" | head -1