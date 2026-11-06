#!/bin/bash
# run_get_chat.sh — fetch the licensed chat data (CPU only, runs alongside the GPU job).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/get_chat_data.py 2>&1 | tail -25
echo
echo "=== on disk ==="
ls -la data/chat/ 2>/dev/null | head -10
df -h "/run/media/yahya/HARD DISK 4" | tail -1