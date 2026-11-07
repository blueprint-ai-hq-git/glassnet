#!/bin/bash
# Launch the seed-2 20k baseline as a transient user service (survives a turn end).
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
UNIT="glassnet-20k-s2"
LOG="lens/runs/DATASCALE-20K-S2-LAUNCH.txt"

{
  echo "=== $(date '+%F %T') launch $UNIT"
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSED: $UNIT is already active - not starting a second copy"
    exit 0
  fi
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
  systemd-run --user --unit="$UNIT" \
    -p MemoryMax=10G -p KillMode=process -p TimeoutStopSec=infinity \
    -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec bash lens/scripts/run_datascale_20k_s2.sh" \
    >> "$LOG" 2>&1 < /dev/null
  echo "systemd-run rc=$?"
  sleep 3
  echo "--- is-active: $(systemctl --user is-active "$UNIT" 2>&1)"
  mp=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null)
  echo "--- MainPID=$mp  cgroup:"; [ -n "$mp" ] && [ "$mp" != "0" ] && cat "/proc/$mp/cgroup" 2>/dev/null
} 2>&1 | tee -a "$LOG"
