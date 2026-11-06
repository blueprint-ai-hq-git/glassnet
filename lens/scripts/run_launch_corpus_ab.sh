#!/bin/bash
# Detached launcher for the corpus A/B chain (waits for the small->big queue, then runs).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_corpus_ab.sh > /dev/null 2>&1 &
sleep 5
echo "chain pid=$(pgrep -f '[r]un_corpus_ab' | head -1)"
echo "queue still running: $(pgrep -f '[r]un_small_big_queue.sh' | head -1)"
echo "current trainer: $(pgrep -f '[t]rain_chat.py' | head -1 | cut -c1-90)"