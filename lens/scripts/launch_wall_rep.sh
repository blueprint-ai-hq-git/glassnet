#!/bin/bash
# Launch the wall-rep block as a TRANSIENT USER SERVICE, not as a child of an agent turn.
#
# WHY (measured twice tonight, 2026-11-06):
#   18:18:02 chain start -> off/seed-2 trainer TERM'd 18:20:00 (exit 143), the minute an agent turn ended.
#   18:27:49 relaunch with persistence -> off/seed-2 trainer TERM'd again at 18:30:15, again the minute a
#   turn ended. In both cases run_wall_rep.sh (setsid'd by the chain) survived and the trainer under it did
#   not. Two arms were lost at two turn boundaries. The project already has the working pattern: the long
#   compare job (stable_vs_plain.py) has been running for hours across many turns because it lives in a
#   systemd user unit (see watchdog.sh:306-309 and launch_resident.sh). This script uses that pattern.
#
# The unit stays after it exits, so `systemctl --user status glassnet-wallrep` is the record, and
# KillMode=process means stopping the unit kills only the main process, not the whole session.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
UNIT="glassnet-wallrep"
LOG="lens/runs/WALL-REP-LAUNCH.txt"

{
  echo "=== $(date '+%F %T') launch $UNIT"
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSED: $UNIT is already active - not starting a second copy"
    systemctl --user status "$UNIT" --no-pager | head -5
    exit 0
  fi
  # a stale transient unit with the same name blocks reuse; reset it only if it is not active
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true

  systemd-run --user --unit="$UNIT" \
    -p MemoryMax=10G -p KillMode=process -p TimeoutStopSec=infinity \
    -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec bash lens/scripts/run_wall_rep.sh" \
    >> "$LOG" 2>&1 < /dev/null
  echo "systemd-run rc=$?"
  sleep 3
  echo "--- is-active: $(systemctl --user is-active "$UNIT" 2>&1)"
  systemctl --user status "$UNIT" --no-pager 2>&1 | sed -n '1,8p'
  echo "--- cgroup of the unit's main process (must NOT be the agent's session scope):"
  mp=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null)
  echo "    MainPID=$mp"
  [ -n "$mp" ] && [ "$mp" != "0" ] && cat "/proc/$mp/cgroup" 2>/dev/null
} 2>&1 | tee -a "$LOG"
