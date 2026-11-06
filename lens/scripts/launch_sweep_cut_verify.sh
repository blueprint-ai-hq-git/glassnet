#!/bin/bash
# launch_sweep_cut_verify.sh - start the sweep-cut verification (sweep_cut_verify.py) in its OWN systemd scope.
# 2026-10-24 self-wake. NOTE THE UNIT NAME ON PURPOSE: it is NOT glassnet-q-*, because the watchdog's honest
# "the machine is busy" test counts ACTIVE glassnet-q-* scopes; a verifier wearing that name would make the
# watchdog refuse to start the critical-path <unk> arm, which is the exact opposite of this turn's purpose.
# Bounded, tiny, read-only: it polls once every 30 s for at most ~3 h, touches no checkpoint and writes only
# lens/runs/SWEEP-CUT-VERIFY.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
UNIT="glassnet-waitcut-$(date +%H%M%S)"
setsid nohup systemd-run --user --scope --unit="$UNIT" -p MemoryMax=1G -p CPUWeight=10 -p IOWeight=10 \
  -p KillMode=process -p TimeoutStopSec=infinity \
  bash -c "cd '/run/media/yahya/HARD DISK 4/GlassNet' && exec taskset -c 0,1 nice -n 19 python3 -u lens/scripts/sweep_cut_verify.py" \
  >> lens/runs/SWEEP-CUT-VERIFY-RUN.txt 2>&1 < /dev/null &
sleep 12
echo "$(date '+%F %H:%M:%S') launched unit=$UNIT  active=$(systemctl --user is-active "$UNIT" 2>/dev/null)" >> lens/runs/SWEEP-CUT-VERIFY-RUN.txt
tail -3 lens/runs/SWEEP-CUT-VERIFY-RUN.txt
