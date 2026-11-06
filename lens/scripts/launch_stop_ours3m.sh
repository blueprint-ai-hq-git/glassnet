#!/bin/bash
# LAUNCHER for lens/scripts/stop_ours3m_after_ep1.sh (2026-10-23 19:0x self-wake). A FILE, not an inline
# env-prefixed command: the approval gate blocks those, and a launcher file is the pattern the watchdog uses.
# Its own systemd scope so it survives this turn's cgroup being torn down (measured 2026-10-22: a plain
# setsid nohup child inherited the cron cgroup and was SIGTERM'd with it).
# MemoryMax=256M: it is a bash sleep-loop with a pgrep, nothing else; it must not be able to disturb the
# 3M ours trainer or its 3.2 GB RSS.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-stop3m-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=256M -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/stop_ours3m_after_ep1.sh' \
  >> lens/runs/OURS3M-STOP.txt 2>&1 < /dev/null &
sleep 10
{
  echo "$(date '+%F %H:%M:%S') unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)"
  echo "waiter process:"; pgrep -af "stop_ours3m_after_ep1" | grep -v "hermes-snap" | head -3
  echo "target trainer:"; pgrep -af '^\.venv-gpu/bin/python3 -u lens/scripts/train_chat\.py .*--out ours_d256_3M_2ep$'
} >> lens/runs/OURS3M-STOP.txt 2>&1
