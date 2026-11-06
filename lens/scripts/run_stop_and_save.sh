#!/bin/bash
# STOP AND SAVE - kills the training work, verifies the GPU really went quiet, and writes a state file
# that says exactly how to resume. Nothing is deleted; every checkpoint stays on disk.
# The two chains now SKIP finished arms and RESUME partial ones, so a stop costs at most the current
# epoch. That resume support was added before this stop for exactly this reason.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
STAMP=$(date '+%Y-%m-%d %H:%M')
STATE=lens/runs/STOP-2026-10-17.md
LOG=/tmp/stop_save.log
exec > "$LOG" 2>&1

echo "=== STOP AND SAVE $STAMP"
BEFORE_GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)
BEFORE_TRAINER=$(pgrep -af "[t]rain_chat" | head -1)
echo "GPU before: $BEFORE_GPU"
echo "trainer before: ${BEFORE_TRAINER:-none}"
ARM=$(echo "$BEFORE_TRAINER" | grep -oE '\-\-out [A-Za-z0-9_]+' | sed 's/--out //')
EP=$(ls -t lens/runs/chat/${ARM}_ep*.npz 2>/dev/null | head -1 | sed 's/.*_ep//; s/\.npz//')
echo "arm at stop: ${ARM:-none} epoch ${EP:-?}"

# wrappers FIRST so no wrapper can start the next arm while we are killing the trainer
for pat in "[r]un_tomorrow" "[r]un_corpus_ab" "[r]un_full_schedule" "[r]un_h_block" "[r]un_launch"; do
  for p in $(pgrep -f "$pat"); do echo "killing wrapper $p ($pat)"; kill "$p"; done
done
sleep 3
for p in $(pgrep -f "[t]rain_chat"); do echo "killing trainer $p"; kill "$p"; done
sleep 6
# a trainer that ignores SIGTERM gets one more, then we report honestly rather than claim success
for p in $(pgrep -f "[t]rain_chat"); do echo "trainer $p still alive, sending SIGKILL"; kill -9 "$p"; done
sleep 3

AFTER_TRAINERS=$(pgrep -c -f "[t]rain_chat")
AFTER_WRAPPERS=$(pgrep -c -f "[r]un_tomorrow|[r]un_corpus_ab|[r]un_full_schedule|[r]un_h_block")
AFTER_GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)
echo "GPU after: $AFTER_GPU"
echo "trainers alive after: $AFTER_TRAINERS   wrappers alive after: $AFTER_WRAPPERS"

NEWEST=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -5)

{
echo "# STOP AND SAVE - $STAMP"
echo
echo "Requested by the user (out of the house). The machine stays on; the heavy work is stopped."
echo "Nothing was deleted. Every checkpoint and every log is still on disk."
echo
echo "## Measured before -> after"
echo "- GPU before: $BEFORE_GPU"
echo "- GPU after:  $AFTER_GPU"
echo "- trainers alive after the stop: $AFTER_TRAINERS (0 means fully stopped)"
echo "- queue wrappers alive after the stop: $AFTER_WRAPPERS (0 means nothing will restart by itself)"
echo
echo "## Interrupted arm"
echo "- arm: ${ARM:-none}  at epoch ${EP:-?} of its schedule"
echo "- the newest checkpoint of that arm is on disk, and the chain now resumes from it"
echo
echo "## Newest checkpoints at stop"
echo '```'
echo "$NEWEST"
echo '```'
echo
echo "## What is still queued (nothing running, all of it resumable)"
echo "- fs_F2 (splitfix d512, 192 ep) - partially trained, resumes"
echo "- fs_F3 (merged d512, 192 ep), fs_F4 (splitfix d256, 192 ep) - not started"
echo "- H-block: H1 reproduce control, H2 no concept supervision, H3 window 64, H4 192 ep, H5 d512 192 ep"
echo
echo "## How to resume (one command, nothing restarts from scratch)"
echo '```'
echo 'bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_resume.sh"'
echo '```'
echo "Finished arms are skipped automatically (their eval is already in the log), partial arms resume"
echo "from their newest checkpoint, and the H-block waits until the GPU is quiet before it starts."
echo
echo "## First numbers already measured before the stop (do not lose them)"
echo "- fs_F1 (merged d256, 192 ep): raw 13.05% -> HONEST 4.62%, text -11.31 WORD SALAD"
echo "- sb_S0 (merged d256, 96 ep):  raw 13.81% -> HONEST 10.21% (best honest so far)"
echo "- sb_S2 (splitfix d256):       raw 36.10% -> HONEST 3.56%"
echo "- sb_S3 (splitfix d512):       raw 38.53% -> HONEST 3.77%"
echo "- sb_B1 (merged d512 + attn):  raw 12.97% -> HONEST 6.04%"
echo
echo "## Reliability while stopped"
echo "- the hourly status digest job is PAUSED so it cannot ping you with nothing-running messages"
echo "- resume it with the cron job id 7ee54a3ab947 when work restarts"
} > "$STATE"

echo
echo "state file written: $STATE"
cat "$STATE"