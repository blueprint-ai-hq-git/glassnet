#!/bin/bash
# Launch the control-token ladder DETACHED: d=64,128,256,512,1024 with the full gate battery at each rung.
# It waits for the running size sweep to finish first, so the two never share the 6 GB card.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_ladder_ctrl.sh > /tmp/ladder_ctrl_detached.log 2>&1 < /dev/null &
echo "control-token ladder launched detached, pid $!"
sleep 3
pgrep -af "[r]un_ladder_ctrl" | cut -c1-60
