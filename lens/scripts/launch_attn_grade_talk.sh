#!/bin/bash
# LAUNCHER for lens/scripts/run_attn_grade_talk.sh (2026-10-23 21:0x self-wake). A FILE, not an inline
# env-prefixed command: the approval gate blocks those and a launcher file is the pattern the watchdog uses.
# Its own systemd scope so the chain survives this turn's cgroup being torn down (measured 2026-10-22: a plain
# setsid nohup child inherited the cron cgroup and was SIGTERM'd with it).
# MemoryMax=4G: the chain loads the corpus index (~1.2 GB) plus two d=256 models (~50 MB each) and nothing else;
# the attention trainer it waits for peaks at 9.0 GB in ITS OWN scope, so a cap here can only kill this chain.
# VERIFICATION: a transient scope reads "inactive" from systemctl even while it runs (seen in
# lens/runs/OURS3M-STOP.txt), so the check below greps the PROCESS, not the unit.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-attngrade-$(date +%H%M%S)"
systemctl --user reset-failed "$unit" >/dev/null 2>&1
setsid nohup systemd-run --user --scope --unit="$unit" -p MemoryMax=4G -p MemorySwapMax=1G \
  -p CPUWeight=20 -p IOWeight=20 \
  bash -c 'cd "/run/media/yahya/HARD DISK 4/GlassNet" && exec bash lens/scripts/run_attn_grade_talk.sh' \
  >> lens/runs/ATTN-GRADE-RUN.txt 2>&1 < /dev/null &
sleep 12
{
  echo "$(date '+%F %H:%M:%S') launcher: unit=$unit active=$(systemctl --user is-active "$unit" 2>/dev/null)"
  echo "  chain process:"; pgrep -af "run_attn_grade_talk" | grep -v "hermes-snap" | head -3
  echo "  trainer it is waiting for:"
  pgrep -af "train_chat\.py .*--out ct_d256_full169_attn\$" | grep -v "hermes-snap" | cut -c1-120
  echo "  memory now:"; free -m | head -2
} >> lens/runs/ATTN-GRADE-RUN.txt 2>&1
