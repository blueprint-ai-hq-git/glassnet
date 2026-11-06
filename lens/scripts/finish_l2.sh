#!/bin/bash
# finish_l2.sh — record the L2 milestone, copy key files, check state.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_l2.py 2>&1 | tail -3
cp STATUS.md REBUILD-PLAN.md /home/yahya/ 2>/dev/null
echo "copied to ~"
echo
echo "=== lens layout so far ==="
find lens -type f -name "*.py" -o -name "*.c" -o -name "*.cu" -o -name "*.so" | sort | sed 's/^/  /'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== jobs ==="
ps -eo pid,cmd | grep -E "g2_scaling|test_l2|lens/" | grep -v grep || echo "  none"