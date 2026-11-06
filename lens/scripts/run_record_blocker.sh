#!/bin/bash
# run_record_blocker.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_blocker.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md REBUILD-PLAN.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== blocker artifacts ==="
ls -la lens/runs/blocker/ 2>/dev/null | tail -10
echo
echo "=== jobs / GPU ==="
ps -eo pid,cmd | grep -E "run_blocker|test_l" | grep -v grep || echo "  nothing running"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader