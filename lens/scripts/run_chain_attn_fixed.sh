#!/bin/bash
# CHAIN 3 — when the splitfix run finishes, run the attention experiment on the FIXED engine
# (batched-matmul bmm3, ~1.6 h instead of 15 h) and report its epoch timing, so the speed fix is
# proven on the real trainer, not just in a microbenchmark.
# The pgrep pattern uses a bracket so it can never match this script's own command line
# (the phantom-match lesson).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while pgrep -f "[t]rain_chat.py .*--out t12_splitfix_d512" >/dev/null 2>&1; do sleep 60; done
echo "splitfix finished at $(date +%H:%M) -- starting attention run on the fixed engine"
bash lens/scripts/run_t12_attn.sh > /tmp/t12_attn_fixed.log 2>&1
echo "attention run finished at $(date +%H:%M)"
grep -E "epoch|ep " /tmp/t12_attn_fixed.log | tail -3