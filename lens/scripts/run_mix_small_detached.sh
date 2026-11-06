#!/bin/bash
# Launch the small smart model run DETACHED. It waits for the combined corpus and for the GPU ladders.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_mix_small.sh > /tmp/mix_small_detached.log 2>&1 < /dev/null &
echo "small smart model run launched detached, pid $!"
sleep 3
pgrep -af "[r]un_mix_small" | cut -c1-60
