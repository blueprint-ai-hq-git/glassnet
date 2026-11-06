#!/bin/bash
# gpu_watch.sh — lightweight GPU health watchdog for gaming sessions.
# Every crontab run: logs VRAM/error-state to runs/gpu_watch_gaming.log and
# writes runs/gpu_watch_alert (content = danger) ONLY when VRAM is nearly
# exhausted or the GPU disappears (the freeze signature). CPU cost ~0.
LOG="/run/media/yahya/HARD DISK 4/GlassNet/runs/gpu_watch_gaming.log"
ALERT="/run/media/yahya/HARD DISK 4/GlassNet/runs/gpu_watch_alert"
STAMP=$(date '+%Y-%m-%d %H:%M:%S')
V=$(timeout 10 nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$V" ]; then
    echo "$STAMP CRITICAL nvidia-smi-failed (GPU hang signature)" >> "$LOG"
    echo "$STAMP CRITICAL: GPU gone (nvidia-smi can't see it) — same signature as the freeze. Roblox must be closed/reboot NOW." > "$ALERT"
    exit 0
fi
USED=$(echo "$V" | cut -d, -f1 | tr -d ' ')
TOT=$(echo "$V" | cut -d, -f2 | tr -d ' ')
echo "$STAMP used=${USED}MiB/${TOT}MiB" >> "$LOG"
if [ "$USED" -gt 5500 ]; then
    echo "$STAMP CRITICAL: VRAM ${USED}MiB — desktop margin gone, gnome-shell can die any second. Close Roblox NOW." > "$ALERT"
elif [ "$USED" -gt 4900 ]; then
    echo "$STAMP WARNING: VRAM ${USED}MiB — climbing, watch it." > "$ALERT"
else
    rm -f "$ALERT"
fi