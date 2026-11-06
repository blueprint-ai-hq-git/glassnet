#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_chat_run1.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== running ==="
ps -eo pid,etime,cmd | grep -E "run_distill|train_chat|run_queue" | grep -v grep | head -4
echo "=== distillation seeds so far ==="
ls -1 lens/runs/ | grep distill