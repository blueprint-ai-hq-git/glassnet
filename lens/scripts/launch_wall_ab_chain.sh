#!/bin/bash
# Launch the WALL-AB experiment THROUGH THE CHAIN, in its own transient unit so it survives the agent turn
# that starts it. Chain = glassnet_chain.sh: it runs the work script, writes the rc, then immediately starts a
# fresh GlassNet agent to do the next piece of work. Nothing here is an inline command chain.
set -u
ROOT="/run/media/yahya/HARD DISK 4/GlassNet"
LABEL="wall_ab"
WORK="$ROOT/lens/scripts/run_wall_ab.sh"
UNIT="glassnet-chain-${LABEL}-$(date +%Y%m%d-%H%M%S)"
out=$(systemd-run --user --unit="$UNIT" --collect -p MemoryMax=10G -p KillMode=process \
        -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
        bash -c "cd \"$ROOT\" && exec nice -n 19 ionice -c3 bash /home/yahya/.local/bin/glassnet_chain.sh \"$WORK\" $LABEL" 2>&1)
rc=$?
stamp="$(date '+%F %H:%M:%S')"
echo "$stamp CHAIN-LAUNCH label=$LABEL unit=$UNIT rc=$rc :: $out" >> "$ROOT/lens/runs/CHAIN-LOG.txt"
echo "rc=$rc unit=$UNIT :: $out"
sleep 3
echo "active=$(systemctl --user is-active "$UNIT" 2>/dev/null) claim=$(ls -1 "$ROOT/lens/runs/GPU-CLAIM" 2>/dev/null || echo none)"
