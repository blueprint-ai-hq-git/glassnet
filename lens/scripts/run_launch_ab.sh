#!/bin/bash
# Launcher for the corpus A/B (detached, so it survives this chat).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_ab_corpus.sh > /dev/null 2>&1 &
sleep 5
echo "ab pid=$(pgrep -f '[r]un_ab_corpus' | head -1)"
echo "trainer: $(pgrep -f '[t]rain_chat.py' | head -1)"