#!/bin/bash
# Launch the data-scale arm as a transient user service (same pattern as launch_wall_rep.sh).
#
# Measured reason, twice tonight: an arm launched inside an agent turn is TERM'd (exit 143) the minute that
# turn ends - 18:20:00 and 18:30:15 both took a trainer with them. The same pattern used for wall_rep
# (systemd-run --user --unit=...) survived every turn end and finished 4/4 arms, so this one uses it too.
set -u
SEED="${1:-0}"
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
UNIT="glassnet-datascale-honest"
[ "$SEED" != "0" ] && UNIT="glassnet-datascale-honest-s$SEED"
# NOT "glassnet-datascale": another agent's inline chain owns that unit (19:05:43) and this agent turn runs
# INSIDE it, so stopping or reusing it would kill this turn.
LOG="lens/runs/DATASCALE-LAUNCH.txt"

{
  echo "=== $(date '+%F %T') launch $UNIT (seed $SEED)"
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSED: $UNIT is already active - not starting a second copy"
    exit 0
  fi
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
  systemd-run --user --unit="$UNIT" \
    -p MemoryMax=10G -p KillMode=process -p TimeoutStopSec=infinity \
    -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec bash lens/scripts/run_datascale_honest.sh $SEED" \
    >> "$LOG" 2>&1 < /dev/null
  echo "systemd-run rc=$?"
  sleep 3
  echo "--- is-active: $(systemctl --user is-active "$UNIT" 2>&1)"
  mp=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null)
  echo "--- MainPID=$mp  cgroup:"; [ -n "$mp" ] && [ "$mp" != "0" ] && cat "/proc/$mp/cgroup" 2>/dev/null
} 2>&1 | tee -a "$LOG"
