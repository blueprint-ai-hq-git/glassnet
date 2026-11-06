#!/bin/bash
# supervisor.sh - keeps the machine WORKING without anyone watching.
#
# The user's complaint (2026-10-19): "it is dead again, just [build] something that will make you active,
# I have to come and tell you every time". The digest reported; the watchdog repaired; but nothing STARTED
# the next job when the previous one finished, so the GPU sat idle until he looked. That is what this fixes.
#
# WHAT IT DOES, every TICK seconds, forever:
#   1. if a training job is running, leave it alone
#   2. if nothing is running and the queue has a job, START it
#   3. if nothing is running and the queue is EMPTY, say so once (not every tick) and keep waiting
#   4. if a job dies before finishing, the queue entry is still there, so it is restarted automatically
#
# The queue is lens/runs/QUEUE.txt, one shell command per line, '#' comments ignored. Finished entries are
# appended to lens/runs/QUEUE-DONE.txt and removed from the queue, so nothing runs twice.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME

TICK=${TICK:-45}
QUEUE=lens/runs/QUEUE.txt
DONEQ=lens/runs/QUEUE-DONE.txt
LOG=lens/runs/SUPERVISOR.txt
IDLE_SAID=0

echo "=== SUPERVISOR started $(date '+%F %H:%M') pid $$" >> "$LOG"

while true; do
  running=$(pgrep -cf "[t]rain_chat")
  if [ "${running:-0}" -gt 0 ]; then
    IDLE_SAID=0
    sleep "$TICK"
    continue
  fi

  # next queued command (skip comments and blanks)
  cmd=$(grep -vE '^\s*#|^\s*$' "$QUEUE" 2>/dev/null | head -1)

  if [ -z "$cmd" ]; then
    if [ "$IDLE_SAID" -eq 0 ]; then
      echo "  $(date '+%H:%M') queue EMPTY and nothing running: idle. Add work to $QUEUE and it starts within ${TICK}s." >> "$LOG"
      IDLE_SAID=1
    fi
    sleep "$TICK"
    continue
  fi

  echo "  $(date '+%H:%M') starting queued job: $cmd" >> "$LOG"
  # remove it from the queue FIRST so a crash cannot loop it forever, then run it in its own scope
  grep -vxF "$cmd" "$QUEUE" > "$QUEUE.tmp" 2>/dev/null && mv "$QUEUE.tmp" "$QUEUE"
  echo "$(date '+%F %H:%M') $cmd" >> "$DONEQ"
  setsid nohup bash -c "$cmd" >> "$LOG" 2>&1 < /dev/null &
  sleep 20
  # wait for the job to actually appear before looping, so the next tick does not double-start it
  for _ in $(seq 1 10); do
    [ "$(pgrep -cf '[t]rain_chat')" -gt 0 ] && break
    sleep 10
  done
done