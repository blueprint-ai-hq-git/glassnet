#!/bin/bash
# Launch the size sweep DETACHED: it trains d=128, 256, 512 and runs the whole gate battery at each size.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_size_sweep.sh > /tmp/size_sweep_detached.log 2>&1 < /dev/null &
echo "size sweep launched detached, pid $!"
sleep 3
pgrep -af "[r]un_size_sweep" | cut -c1-60
