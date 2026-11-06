#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_char.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== the queue, in order ==="
ps -eo pid,etime,cmd | grep -E "run_queue|run_optimise|train_chat|run_distill|run_chat_char" | grep -v grep | head -8
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== disk ==="
df -h "/run/media/yahya/HARD DISK 4" | tail -1