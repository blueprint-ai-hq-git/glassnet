#!/bin/bash
# Launch the cosmopedia download detached (see the glassnet skill on why not a Hermes background job).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_get_cosmopedia.sh > /tmp/cosmopedia_launch.log 2>&1 < /dev/null &
echo "cosmopedia download launched detached, pid $!"
sleep 3
pgrep -af "[r]un_get_cosmopedia" | cut -c1-60