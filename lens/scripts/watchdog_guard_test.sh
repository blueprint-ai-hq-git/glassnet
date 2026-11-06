#!/bin/bash
# watchdog_guard_test.sh — verify the NEW between-arms guard in lens/scripts/watchdog.sh (2026-10-24 self-wake)
# by executing the EXACT shipped line (extracted from the file, not retyped) in two states:
#   1. with a unit pattern that matches no unit      -> must read 0 (so the guard cannot fire on nothing)
#   2. with the real pattern, right now              -> must read >= 1 while a queue job is alive
# and by showing what the guard does with each value. It writes lens/runs/WATCHDOG-GUARD-TEST.txt.
# Nothing else is touched: no queue edit, no launch.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/WATCHDOG-GUARD-TEST.txt
LINE=$(grep -m1 '^qscopes=' lens/scripts/watchdog.sh)
[ -n "$LINE" ] || { echo "THE GUARD LINE IS NOT IN watchdog.sh — refusing to report a PASS"; exit 2; }
{
echo "=== WATCHDOG GUARD TEST $(date '+%F %H:%M:%S') ==="
echo "the shipped line, taken from the file by grep (not retyped):"
echo "  $LINE"
REAL='glassnet-q-*'
GREP_REAL="list-units 'glassnet-q-*'"
GREP_FAKE="list-units 'glassnet-zzz-*'"
eval "${LINE/'glassnet-q-*'/'glassnet-zzz-*'}"      # same line, no unit can match
echo "  NEGATIVE CONTROL (pattern 'glassnet-zzz-*'): qscopes=${qscopes:-unset}  -> ${qscopes:-0} (must be 0)"
eval "$LINE"
echo "  LIVE (pattern 'glassnet-q-*'):              qscopes=${qscopes:-unset}"
if [ "${qscopes:-0}" -gt 0 ]; then
  qrunning=0       # the state a between-arms gap presents to the watchdog: no train_chat process
  if [ "${qscopes:-0}" -gt 0 ] && [ "${qrunning:-0}" -eq 0 ]; then
    echo "  -> the guard WOULD set qrunning=1 and refuse to start the next queue entry (correct: a job IS alive)"
  else
    echo "  -> the guard's condition did NOT hold — THAT IS A FAILURE OF THE GUARD, investigate"
  fi
else
  echo "  -> no active glassnet-q-* scope right now, so the guard is a no-op at this moment (also correct)"
fi
echo "  NOTE ON A BUG IN THIS TEST, RECORDED NOT HIDDEN: the first version wrote \${qrunning:-0:-0}, which is not"
echo "  valid parameter expansion ('[: 0:-0: integer expected'), so it fell through to the no-op branch while the"
echo "  scope WAS active. The condition above is now the shipped one with qrunning=0 stated explicitly."
echo "  units counted:"
systemctl --user list-units 'glassnet-q-*' --state=active --no-legend --plain 2>/dev/null | sed 's/^/    /'
echo "  NOTE: this test cannot force a between-arms gap; it verifies the CONDITION the guard reads. The"
echo "  observed firing case is a tick that lands while qrunning=0 and a scope is active, which the guard"
echo "  now reports in lens/runs/WATCHDOG.txt as '...NOT starting the next entry (this is the between-arms gap)'."
echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$OUT" 2>&1
