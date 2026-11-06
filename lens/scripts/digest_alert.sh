#!/bin/bash
# digest_alert.sh - hourly watchdog that SPEAKS ONLY WHEN THERE IS SOMETHING TO SAY.
# Prints nothing (so nothing is sent) unless one of these is true:
#   * an arm finished since the last report
#   * a trainer is running but has not written a checkpoint for more than 25 minutes (STALL)
#   * nothing is running at all while a queue wrapper is still alive (a broken chain)
#   * it has been more than 2 hours since the last report (so silence never lasts long)
# Used by the hourly cron job with no_agent, so the text is delivered exactly as printed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
STATE=lens/runs/.digest_state
DIGEST=$(bash lens/scripts/digest.sh 2>/dev/null)

# a fingerprint of "which arms have finished" plus the newest checkpoint age
FIN=$(echo "$DIGEST" | grep -c "honest\|single number")
NEWEST=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1)
AGE=99999
if [ -n "$NEWEST" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$NEWEST") ))
fi
TRAINER=$(pgrep -f "[t]rain_chat" | head -1)
CHAIN=$(pgrep -f "[r]un_tomorrow|[r]un_h_block|[r]un_corpus_ab|[r]un_full_schedule" | head -1)
FP="$FIN:$AGE:$TRAINER:$CHAIN"

LAST_FP=""; LAST_T=0
[ -f "$STATE" ] && . "$STATE"

WHY=""
[ "$FIN" != "$LAST_FP" ] && WHY="a new arm finished"
if [ -n "$TRAINER" ] && [ "$AGE" -gt 1500 ]; then
  WHY="STALL: a trainer is running but has written no checkpoint for $((AGE/60)) minutes"
fi
if [ -z "$TRAINER" ] && [ -n "$CHAIN" ]; then
  WHY="BROKEN CHAIN: a queue script is alive but no trainer is running"
fi
if [ $(( $(date +%s) - LAST_T )) -gt 7200 ]; then
  WHY="${WHY:-no report for over 2 hours}"
fi

if [ -z "$WHY" ]; then
  exit 0
fi

echo "GlassNet status  $(date '+%a %H:%M')   ($WHY)"
echo
echo "$DIGEST"
echo
FAILS=$(grep -cE "\[FAIL\]" /tmp/product_check_selfcheck.log 2>/dev/null)
echo "product check (last run): ${FAILS:-0} failed checks    bugs logged: $(grep -c '^## ' BUGS.md 2>/dev/null)"

printf 'LAST_FP="%s"\nLAST_T=%s\n' "$FIN" "$(date +%s)" > "$STATE"