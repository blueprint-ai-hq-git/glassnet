#!/bin/bash
# Launch the BIG rungs DETACHED: d=1536, 2048, 3072 (about 21M / 28M / 41M parameters, inside the 50M cap).
# Serialized behind every other GPU job by the runner itself.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_ladder_big.sh > /tmp/ladder_big_detached.log 2>&1 < /dev/null &
echo "big rungs launched detached, pid $!"
sleep 3
pgrep -af "[r]un_ladder_big" | cut -c1-60
