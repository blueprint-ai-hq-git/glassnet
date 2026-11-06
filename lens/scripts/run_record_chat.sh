#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_chat.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== what is running ==="
ps -eo pid,etime,cmd | grep -E "run_distill|run_chat_real|train_chat" | grep -v grep | head -4