#!/bin/bash
# run_record_crossover.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_crossover.py 2>&1 | tail -3
cp STATUS.md LOG.md REBUILD-PLAN.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== gates on record ==="
ls lens/tests/test_l*.py 2>/dev/null | sed 's/^/  /'
echo
echo "=== GPU / jobs ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -eo pid,cmd | grep -E "test_l2|g2_scaling" | grep -v grep || echo "  nothing running"