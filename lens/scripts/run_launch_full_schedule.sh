#!/bin/bash
# Detached launcher for the full-schedule chain (waits for the corpus arms, then runs the 192-epoch arms).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_full_schedule.sh > /dev/null 2>&1 &
sleep 6
echo "full-schedule chain pid=$(pgrep -f '[r]un_full_schedule' | head -1)"
echo "corpus chain pid=$(pgrep -f '[r]un_corpus_ab' | head -1)"
echo "small->big queue pid=$(pgrep -f '[r]un_small_big_queue' | head -1)"
echo "current trainer: $(pgrep -af '[t]rain_chat.py' | head -1 | cut -c1-80)"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"