#!/bin/bash
# stop_all.sh - stop every GlassNet job cleanly for a shutdown.
#
# Order matters: stop the training FIRST (so it can finish writing its checkpoint), then the loops that would
# restart it. The last checkpoint of every run is on disk before anything else happens.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1

echo "=== STOPPING $(date '+%F %H:%M')"
echo "before: training procs = $(pgrep -cf '[t]rain_chat'), scopes = $(systemctl --user list-units --type=scope --no-pager 2>/dev/null | grep -ci glassnet)"

# 1. training: SIGTERM first so the process can flush; the trainer writes a checkpoint every epoch
pkill -TERM -f "[t]rain_chat" 2>/dev/null
sleep 8
# 2. the detached helper scripts that would relaunch or report
pkill -f "[r]un_eval_clean" 2>/dev/null
pkill -f "[s]upervisor.sh" 2>/dev/null
pkill -f "[w]atchdog.sh" 2>/dev/null
# 3. any leftover scopes
for u in $(systemctl --user list-units --type=scope --no-pager 2>/dev/null | grep -i glassnet | awk '{print $1}'); do
  systemctl --user stop "$u" 2>/dev/null
done
sleep 3

echo "after: training procs = $(pgrep -cf '[t]rain_chat'), scopes = $(systemctl --user list-units --type=scope --no-pager 2>/dev/null | grep -ci glassnet)"
echo "newest checkpoint: $(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1 | xargs -n1 basename)"
echo "checkpoints on disk: $(ls lens/runs/chat/*.npz 2>/dev/null | wc -l)"
echo "disk free: $(df -h . | tail -1 | awk '{print $4}')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
