#!/bin/bash
# CHAIN 4 — when the attention run finishes, start the merged-corpus run automatically.
# Bracket pattern in pgrep so this script can never match its own command line.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while pgrep -f "[t]rain_chat.py .*--out t12_attn_d512" >/dev/null 2>&1; do sleep 60; done
echo "attention run finished at $(date +%H:%M) -- starting the merged-corpus run"
bash lens/scripts/run_t12_merged.sh > /tmp/t12_merged.log 2>&1
echo "merged run finished at $(date +%H:%M)"