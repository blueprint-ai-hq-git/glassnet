#!/bin/bash
# run_record_tests.sh — record both suites, then confirm the machine is idle.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_tests.py 2>&1 | tail -3
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
echo
echo "=== GlassNet jobs still running (should be none) ==="
ps -eo pid,etime,cmd | grep -E "t3_models|f_train|f_audit|b_grid|test_" | grep -v grep || echo "  none"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== STATUS.md numbered lines ==="
grep -cE "^[0-9]+\." STATUS.md