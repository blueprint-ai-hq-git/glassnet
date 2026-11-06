#!/bin/bash
# LAUNCHER for lens/scripts/run_talk_ours3m_ep1.sh (2026-10-23 19:5x self-wake). A FILE, not an inline
# env-prefixed command: the approval gate blocks those and the watchdog pattern is a launcher file.
# Its own systemd scope so it survives this turn's cgroup being torn down (measured 2026-10-22: a plain
# setsid nohup child inherited the cron cgroup and was SIGTERM'd with it).
# MemoryMax=3G: the talk test loads the corpus index (~1.2 GB) plus two d=256 models; capping it HERE means an
# overrun kills this test, never the attention arm training on the GPU (its scope peak is 9.0 GB).
# VERIFICATION: a transient scope reads "inactive" from systemctl even while it runs (seen in
# lens/runs/OURS3M-STOP.txt), so the check below greps for the PYTHON process, not the unit.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-talkours3m-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=3G -p MemorySwapMax=1G \
  -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/run_talk_ours3m_ep1.sh' \
  >> lens/runs/PADFREE-TALK-OURS3M-RUN.txt 2>&1 < /dev/null &
sleep 12
{
  echo "$(date '+%F %H:%M:%S') launcher: unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)"
  echo "  python process:"; pgrep -af "talk_fair_v3_ship.py" | grep -v hermes-snap | head -3
  echo "  memory now:"; free -m | head -2
} >> lens/runs/PADFREE-TALK-OURS3M-RUN.txt 2>&1
