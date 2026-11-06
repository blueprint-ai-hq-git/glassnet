#!/bin/bash
# CHAIN 2 — wait for the attention run to finish, then run the splitfix experiment (full
# corpus + leak-free val) automatically, then report. Same cheap 60s check loop on the actual
# trainer process.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while pgrep -f "train_chat.py .*--out t12_attn_d512" >/dev/null 2>&1; do sleep 60; done
echo "attention run finished at $(date +%H:%M) -- starting splitfix"
bash lens/scripts/run_t12_splitfix.sh > /tmp/t12_splitfix.log 2>&1
echo "splitfix done at $(date +%H:%M)"