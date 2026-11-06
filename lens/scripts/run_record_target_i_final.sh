#!/bin/bash
# run_record_target_i_final.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_target_i_final.py 2>&1 | tail -3
cp TARGET-LEDGER.md STATUS.md /home/yahya/ 2>/dev/null
echo "copied"
echo
echo "=== nothing running? ==="
ps -eo pid,cmd | grep -E "g2_scaling|target_i|test_l" | grep -v grep || echo "  nothing running"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader