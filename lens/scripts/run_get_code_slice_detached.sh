#!/bin/bash
# Launch the code-slice download DETACHED (see the glassnet skill: Hermes-owned background jobs get
# reaped when the agent's worker scope is torn down, which killed the first knowledge download).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_get_code_slice.sh > /tmp/code_slice_launch.log 2>&1 < /dev/null &
echo "code slice download launched detached, pid $!"
sleep 3
pgrep -af "[r]un_get_code_slice" | cut -c1-60
