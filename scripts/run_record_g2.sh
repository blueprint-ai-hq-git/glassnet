#!/bin/bash
# run_record_g2.sh — record the Target G2 verdict.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_g2.py 2>&1 | tail -3
cp STATUS.md TARGET-G2-RESULTS.md /home/yahya/
echo "=== copied ==="
echo "=== machine ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -eo pid,cmd | grep -E "g2_scaling|g_loop" | grep -v grep || echo "  nothing of mine running"