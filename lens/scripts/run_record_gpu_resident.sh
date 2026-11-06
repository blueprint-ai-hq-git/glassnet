#!/bin/bash
# run_record_gpu_resident.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_gpu_resident.py 2>&1 | tail -2
cp STATUS.md REBUILD-PLAN.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== scale test still running? ==="
ps -eo pid,etime,cmd | grep test_l2_scale | grep -v grep | head -2 || echo "  finished"