#!/bin/bash
# feed_queue.sh - the machine gives itself the next job.
#
# The user's requirement (2026-10-19): "set up yourself to auto startup, check every few minutes, and CONTINUE
# -- I go offline, I give you the work, and you do it". A queue that runs dry and waits for a human is exactly
# the failure he described. So when QUEUE.txt is empty, the next RUN line from PLAN.txt is moved into it, and
# the watchdog (every 5 minutes) starts it. The machine now has work for hours without anyone present.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
Q=lens/runs/QUEUE.txt
P=lens/runs/PLAN.txt
queued=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' "$Q" 2>/dev/null)
queued=${queued:-0}
# NOTE: the old form was `... || echo 0`, which on an empty queue produced TWO lines ('0' from grep
# and '0' from the fallback) and made the next test print '[ : 0\n0: integer expected'. Harmless but
# noisy, and a noisy script is one nobody trusts.
if [ "${queued:-0}" -gt 0 ]; then
  exit 0
fi
# a RUN line whose command is already queued or already finished is marked STARTED and skipped, so the same
# experiment can never be launched twice by the feeder
while read -r line; do
  cmd0=${line#RUN }
  if grep -qxF "$cmd0" "$Q" 2>/dev/null || grep -qF "$cmd0" lens/runs/QUEUE-DONE.txt 2>/dev/null; then
    sed -i "s|^RUN $(printf '%s' "$cmd0" | sed 's/[\/&]/\\&/g')$|STARTED $(date '+%F %H:%M') $cmd0|" "$P" 2>/dev/null
  fi
done < <(grep -E '^RUN ' "$P" 2>/dev/null)

next=$(grep -E '^RUN ' "$P" 2>/dev/null | head -1)
if [ -z "$next" ]; then
  echo "$(date '+%F %H:%M') PLAN EXHAUSTED: no RUN lines left; the agent must add work (CODE lines pending)" \
    >> lens/runs/FEEDER.txt
  exit 0
fi
cmd=${next#RUN }
echo "$cmd" >> "$Q"
# mark the plan line as started (not deleted -- the plan stays a record of what was intended)
sed -i "0,/^RUN $(printf '%s' "$cmd" | sed 's/[\/&]/\\&/g')$/s//STARTED $(date '+%F %H:%M')/" "$P" 2>/dev/null
echo "$(date '+%F %H:%M') fed: $cmd" >> lens/runs/FEEDER.txt
