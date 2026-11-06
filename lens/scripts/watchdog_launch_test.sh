#!/bin/bash
# watchdog_launch_test.sh — DOES THE FIXED LAUNCH BLOCK START EXACTLY ONE COPY?  (2026-10-25 self-wake)
#
# WHY THIS EXISTS: lens/scripts/watchdog.sh section 0 carried a stray `bash -c ... &` line that was NOT behind
# an `else`, so EVERY queued line was launched TWICE — once as a transient service and once in the tick's own
# cgroup. Measured at 2026-10-25 10:30:38 on run_tied_w64.sh (two identical trainers, pids 4020503 and
# 4020510). The line is removed; this test reproduces the launch block VERBATIM with a HARMLESS command
# (`sleep 25`) so the claim "one launch, one copy" is measured instead of asserted.
#
# It does NOT touch the queue, does not start a trainer, and stops the test unit at the end. It writes
# lens/runs/WATCHDOG-LAUNCH-TEST.txt. Its unit name starts with glassnet-q- like the real launcher's, so it is
# visible to the watchdog's busy guard for the ~20 s it lives — the same courtesy the real launcher gets.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/WATCHDOG-LAUNCH-TEST.txt
unit="glassnet-q-testlaunch-$(date +%H%M%S)"
qnext="sleep 25"

{
  echo "=============================================================================================="
  echo "WATCHDOG LAUNCH TEST — one launch must produce exactly ONE copy   $(date '+%F %H:%M:%S')"
  echo "the command launched is HARMLESS ('$qnext'); no trainer, no queue change."
  echo "=============================================================================================="
  systemctl --user reset-failed "$unit" >/dev/null 2>&1
  # ---- the launch block, copied from lens/scripts/watchdog.sh section 0 with the fixed line set
  systemd-run --user --unit="$unit" --collect -p MemoryMax=10G -p KillMode=process \
    -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" \
    >> lens/runs/SUPERVISOR.txt 2>&1 < /dev/null
  sleep 15
  active=$(systemctl --user is-active "$unit" 2>/dev/null)
  copies=$(pgrep -cf "[s]leep 25")
  units=$(systemctl --user list-units 'glassnet-q-testlaunch-*' --all --no-legend --plain 2>/dev/null | grep -c .)
  echo "  unit               : $unit"
  echo "  is-active          : ${active:-<none>}"
  echo "  live copies of '$qnext' : ${copies:-0}   (exactly 1 = PASS, 2 or more = THE DUPLICATE IS STILL THERE)"
  echo "  testlaunch units   : ${units:-0}"
  if [ "${copies:-0}" -eq 1 ] && [ "${active:-x}" = "active" ]; then
    echo "  VERDICT: PASS — one launch, one copy, in its own transient unit."
  else
    echo "  VERDICT: FAIL — copies=${copies:-0} active=${active:-<none>}. Do NOT trust the launcher."
  fi
  systemctl --user stop "$unit" >/dev/null 2>&1
  sleep 2
  echo "  clean-up: test unit stopped (is-active now $(systemctl --user is-active "$unit" 2>/dev/null || echo gone))"
  echo "  NOTE: this proves the launch block starts ONE copy. It does not and cannot prove the queue's own"
  echo "  bookkeeping; that is verified on the next real start by reading QUEUE-DONE.txt for a single line."
  echo "=============================================================================================="
} >> "$OUT" 2>&1
tail -14 "$OUT"
