#!/bin/bash
# CHAIN — wait for the clean-data training to finish, then immediately run the attention
# experiment (the queue, without me polling). Checks the TRAINER process specifically, not a
# bare pgrep pattern (the phantom-wrapper lesson), and sleeps 60s between checks so it costs
# nothing. If the clean run has already exited, this starts the attention run straight away.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while pgrep -f "train_chat.py .*--out t12_clean_d512" >/dev/null 2>&1; do sleep 60; done
echo "clean-data run finished at $(date +%H:%M) -- starting the attention experiment"
bash lens/scripts/run_t12_attn.sh > /tmp/t12_attn.log 2>&1
echo "chain done at $(date +%H:%M)"