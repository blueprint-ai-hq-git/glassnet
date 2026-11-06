#!/bin/bash
# run_record_explain_limit.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_explain_limit.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "copied"
echo
echo "=== chain still running? ==="
ps -eo pid,etime,cmd | grep -E "run_chain|run_scaling|run_distill|run_blocker" | grep -v grep | head -4 \
  || echo "  finished"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader