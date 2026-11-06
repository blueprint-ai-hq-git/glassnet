#!/bin/bash
# Launcher: stop the long overnight supervisor (it would otherwise start the 3-hour merged run at
# ~21:50 and the ~10-hour matrix after it, right as the PC is being powered down) and start the
# short NIGHT FINISH job detached, so it survives this chat session.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
pkill -f "[r]un_overnight_supervisor" && echo "old supervisor stopped" || echo "old supervisor not running"
sleep 2
setsid nohup bash lens/scripts/run_night_finish.sh > /dev/null 2>&1 &
sleep 5
NF=$(pgrep -f "[r]un_night_finish" | head -1)
echo "night-finish pid=${NF:-NONE}"
ps -o pid,ppid,sid,cmd -p "${NF:-1}" 2>/dev/null | tail -1
head -2 /tmp/night_finish.log 2>/dev/null
echo "--- attention trainer still alive:"; pgrep -f "[t]rain_chat.py .*t12_attn" | head -1
ls -t lens/runs/chat/t12_attn_d512_ep*.npz | head -1