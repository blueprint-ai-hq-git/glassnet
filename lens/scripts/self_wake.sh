#!/bin/bash
# SELF-WAKE, REBUILT AS A SCRIPT (2026-10-31). It used to be an LLM job and died twice: HTTP 410 "model
# deepseek-v4-flash has been deprecated" (silent for 3 days) then HTTP 400 "Model is unavailable". A supervisor
# that needs a model to be up is not a supervisor. This version does the same checks with no model at all:
# start queued work, verify the machine, record the real numbers. It prints ONLY when it had to act or found a
# problem, so it stays quiet when everything is healthy.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
acted=""
tr=$(pgrep -cf "venv-gpu/bin/python3 -u lens/scripts/train_chat"); tr=${tr:-0}
q=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null); q=${q:-0}
pl=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/PLAN.txt 2>/dev/null); pl=${pl:-0}
# 1. nothing running but work exists -> start it
if [ "$tr" -eq 0 ] && { [ "$q" -gt 0 ] || [ "$pl" -gt 0 ]; }; then
  before=$tr
  bash lens/scripts/watchdog.sh >/dev/null 2>&1
  sleep 20
  tr2=$(pgrep -cf "venv-gpu/bin/python3 -u lens/scripts/train_chat"); tr2=${tr2:-0}
  [ "$tr2" -gt 0 ] && acted="${acted}started queued work (was idle); " || acted="${acted}TRIED to start work but nothing came up; "
fi
# 2. live view down -> bring it back
live=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8770/ 2>/dev/null)
if [ "$live" != "200" ]; then
  systemctl --user restart glassnet-liveview >/dev/null 2>&1; sleep 5
  live2=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8770/ 2>/dev/null)
  [ "$live2" = "200" ] && acted="${acted}restarted the live view; " || acted="${acted}live view still down; "
fi
# 3. record this check in the state file the next session reads first
{ echo "## $(date '+%F %H:%M') self-wake"; echo "trainers=$tr queued=$q plan=$pl live=${live} acted: ${acted:-nothing needed}"; } >> lens/runs/AGENT-INBOX.md
# 4. speak only if it acted or something is wrong
if [ -n "$acted" ]; then echo "GLASSNET self-wake $(date '+%H:%M'): $acted"; fi
if [ "$tr" -eq 0 ] && [ "$q" -eq 0 ] && [ "$pl" -eq 0 ]; then
  echo "GLASSNET self-wake $(date '+%H:%M'): nothing running and NO queued or planned work - new experiments must be written."
fi
exit 0
