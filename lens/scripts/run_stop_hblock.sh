#!/bin/bash
# Stop the H-block chain and its H1 trainer: it started while the small/big queue was still training
# (the wait loop matched 'run_full_schedule', but the live wrapper is 'run_tomorrow'), so two trainers
# ended up sharing the GPU. One trainer at a time is the rule. H1 restarts later (with --resume if the
# newest checkpoint is intact); its epoch checkpoints stay on disk either way.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
for p in $(pgrep -f "[r]un_h_block"); do echo "killing wrapper $p"; kill "$p"; done
sleep 2
for p in $(pgrep -f "[t]rain_chat.py.*h_H1_merged_honest"); do echo "killing H1 trainer $p"; kill "$p"; done
sleep 4
echo "--- remaining trainers:"; pgrep -af "[t]rain_chat" | cut -c1-80
echo "--- H1 checkpoints kept:"; ls -t lens/runs/chat/h_H1_*.npz 2>/dev/null | head -3
echo "--- GPU after:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader