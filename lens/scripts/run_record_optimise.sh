#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_optimise.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== queue order (each waits for the previous) ==="
ps -eo pid,etime,cmd | grep -E "run_queue|run_optimise|train_chat|run_distill" | grep -v grep | head -6