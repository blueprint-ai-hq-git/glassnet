#!/bin/bash
# Launch the turn-end gate DETACHED from the agent's process tree.
# Why: on 2026-10-18 the trainer was killed mid-run when systemd stopped the Hermes worker scope
# (syslog: "Stopped hermes-worker-proc_8576d38ba2b6.scope"). Long GPU runs must not live inside that
# scope, so this uses setsid + nohup and its own log file.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_turnend_gate.sh > /tmp/turnend_gate_detached.log 2>&1 < /dev/null &
echo "launched detached, pid $!"
sleep 3
pgrep -af "[t]rain_chat" | cut -c1-70
