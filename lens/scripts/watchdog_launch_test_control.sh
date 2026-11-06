#!/bin/bash
# watchdog_launch_test_control.sh — THE SAME TEST WITH THE OLD LAUNCH BLOCK, AS A CONTROL.
# Purpose: prove the removed stray line WAS the duplicate, not something else. With the pre-change block
# (systemd-run FOLLOWED BY an unconditional `bash -c ... &`) one launch must produce TWO live copies; with the
# fixed block (watchdog_launch_test.sh) it produces ONE. Both launch a harmless `sleep` — no trainer, no queue
# change — and both stop their units afterwards.
# Writes lens/runs/WATCHDOG-LAUNCH-TEST-CONTROL.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/WATCHDOG-LAUNCH-TEST-CONTROL.txt
unit="glassnet-q-testlaunchctl-$(date +%H%M%S)"
qnext="sleep 27"

{
  echo "=============================================================================================="
  echo "WATCHDOG LAUNCH TEST — CONTROL, WITH THE PRE-CHANGE BLOCK   $(date '+%F %H:%M:%S')"
  echo "pre-change copy of the launcher: lens/runs/_tmp/watchdog_prefix_20261025-1032.sh (the stray line is"
  echo "the one quoted in the fix comment). The command launched is HARMLESS ('$qnext')."
  echo "=============================================================================================="
  systemctl --user reset-failed "$unit" >/dev/null 2>&1
  systemd-run --user --unit="$unit" --collect -p MemoryMax=10G -p KillMode=process \
    -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" \
    >> lens/runs/SUPERVISOR.txt 2>&1 < /dev/null
    bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" >> lens/runs/SUPERVISOR.txt 2>&1 < /dev/null &
  sleep 15
  copies=$(pgrep -cf "[s]leep 27")
  echo "  unit            : $unit"
  echo "  live copies of '$qnext' : ${copies:-0}"
  if [ "${copies:-0}" -ge 2 ]; then
    echo "  VERDICT: CONTROL CONFIRMS THE BUG — the pre-change block started ${copies} copies of one launch."
  else
    echo "  VERDICT: CONTROL DOES NOT REPRODUCE — ${copies:-0} copy(ies); the duplicate must have another cause."
  fi
  systemctl --user stop "$unit" >/dev/null 2>&1
  pkill -f "[s]leep 27"
  sleep 2
  echo "  clean-up: test unit stopped, stray copies killed (remaining: $(pgrep -cf '[s]leep 27'))"
  echo "=============================================================================================="
} >> "$OUT" 2>&1
tail -10 "$OUT"
