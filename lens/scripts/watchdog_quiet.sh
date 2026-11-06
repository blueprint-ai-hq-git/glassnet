#!/bin/bash
# watchdog_quiet.sh - the scheduled watchdog. Silent when everything is fine, speaks when it FIXED something.
#
# Scheduled jobs deliver stdout verbatim and empty stdout sends nothing, so this wrapper prints only when
# the watchdog actually acted or found something that needs the user's attention. That is the difference
# between "staying updated" and spamming a chat every 30 minutes with the same healthy status.
#
# The user's rule (2026-10-18): "if you find any problem you can get solve it instead of irritating me
# afterwards that it failed and we are late" - so the watchdog repairs first and reports the repair, not the
# symptom.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME

REPORT=$(bash lens/scripts/watchdog.sh 2>&1)

# speak only if it did something, or if something is genuinely wrong
if echo "$REPORT" | grep -qE "FOUND AND FIXED|DISK LOW|OOM kill|relaunching|stalled"; then
  echo "GlassNet watchdog $(date '+%F %H:%M')"
  echo
  echo "$REPORT" | grep -E "^  - |FOUND AND FIXED|found nothing|master run:|newest checkpoint:|disk free:" | tail -12
  echo
  echo "Full history: lens/runs/WATCHDOG.txt"
fi
# healthy ticks stay silent on purpose