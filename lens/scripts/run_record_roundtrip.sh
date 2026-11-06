#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_roundtrip.py 2>&1 | tail -4
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== what is running now ==="
ps -eo pid,etime,cmd | grep -E "train_chat|run_distill|run_optimise|run_queue" | grep -v grep | head -5
echo "=== results so far ==="
ls -1 lens/runs/chat/ 2>/dev/null | head -12