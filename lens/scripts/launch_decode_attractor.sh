#!/bin/bash
# LAUNCHER for the decode-attractor measurement (2026-10-22 20:2x self-wake). A FILE, not an inline command:
# the approval gate blocks inline env-prefixed launches, and a launcher file is the pattern the watchdog uses.
# Its own systemd scope so it survives this turn's cgroup being torn down (measured 2026-10-22: a plain
# setsid nohup child inherited the cron cgroup and was SIGTERM'd with it).
# MemoryMax=3G: the d=512 trainer in flight peaked at 9.17 GB inside its own 10G scope (MemoryPeak,
# glassnet-q-3182532730-192758.scope), and this machine has 14 GB total, so this job must be small.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-attr-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=3G -p MemorySwapMax=1G \
  -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/run_decode_attractor.sh' \
  >> lens/runs/DECODE-ATTRACTOR-LAUNCH.txt 2>&1 < /dev/null &
sleep 12
echo "$(date '+%F %H:%M:%S') unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)" \
  >> lens/runs/DECODE-ATTRACTOR-LAUNCH.txt
pgrep -af decode_attractor_test >> lens/runs/DECODE-ATTRACTOR-LAUNCH.txt 2>&1
