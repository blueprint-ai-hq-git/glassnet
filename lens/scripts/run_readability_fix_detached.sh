#!/bin/bash
# Launch the readability sweep DETACHED (outside the agent's process tree). It waits for the GPU itself.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_readability_fix.sh > /tmp/readability_fix_detached.log 2>&1 < /dev/null &
echo "readability sweep queued detached, pid $!"
sleep 3
pgrep -af "[r]un_readability_fix" | cut -c1-70