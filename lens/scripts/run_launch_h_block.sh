#!/bin/bash
# Detached launcher for the H-block (waits for the running chains, then runs 5 arms).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_h_block.sh > /dev/null 2>&1 &
sleep 6
echo "H-block pid=$(pgrep -f '[r]un_h_block' | head -1)"
echo "chains still ahead of it:"
pgrep -af "[r]un_small_big_queue|[r]un_corpus_ab|[r]un_full_schedule" | cut -c1-70
echo "current GPU arm: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"