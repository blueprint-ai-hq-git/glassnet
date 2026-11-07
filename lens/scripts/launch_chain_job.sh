#!/bin/bash
# Launch a glassnet CHAIN job so that BOTH the work and the wake survive the agent's turn ending.
#
# WHY (measured twice today): glassnet_chain.sh runs the work, then wakes a fresh agent when it exits. That design
# is right, but if the chain itself is started from an agent's terminal it is a child of that turn and is SIGTERMed
# when the turn ends - the 8M arm started 18:13:41 died at 18:15:50 with rc 143 after saving an interrupt
# checkpoint. Putting the CHAIN under a transient user unit fixes both halves: the trainer keeps running and the
# finish still wakes an agent.
#
# Usage: launch_chain_job.sh <work-script-relative-to-repo> <label>
#   e.g. launch_chain_job.sh lens/scripts/run_datascale_8m_nogate.sh datascale_8m
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
WORK="${1:?need a work script}"
LABEL="${2:?need a short label}"
UNIT="glassnet-chain-${LABEL}"
LOG="lens/runs/CHAIN-LAUNCH-${LABEL}.txt"

[ -f "$WORK" ] || { echo "REFUSED: $WORK is not a file (pass a script path, not a command line)" | tee -a "$LOG"; exit 2; }
bash -n "$WORK" 2>/dev/null || { echo "REFUSED: $WORK does not parse as bash (a binary or a broken script)" | tee -a "$LOG"; exit 2; }

{
  echo "=== $(date '+%F %T') launch chain label=$LABEL work=$WORK unit=$UNIT"
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSED: $UNIT is already active - one chain job per label at a time"
    exit 0
  fi
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
  systemd-run --user --unit="$UNIT" \
    -p MemoryMax=14G -p KillMode=process -p TimeoutStopSec=infinity \
    -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd '$PWD' && exec bash /home/yahya/.local/bin/glassnet_chain.sh '$WORK' '$LABEL'" \
    >> "$LOG" 2>&1 < /dev/null
  echo "systemd-run rc=$?"
  sleep 4
  echo "--- is-active: $(systemctl --user is-active "$UNIT" 2>&1)"
  mp=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null)
  echo "--- MainPID=$mp  cgroup: $(cat "/proc/$mp/cgroup" 2>/dev/null | cut -c1-80)"
} 2>&1 | tee -a "$LOG"
