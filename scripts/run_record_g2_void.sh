#!/bin/bash
# run_record_g2_void.sh — record the void sweep and the two new gates.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_g2_void.py 2>&1 | tail -3
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
echo "=== G2 corrected sweep ==="
ps -eo pid,etime,pcpu,cmd | grep g2_scaling | grep -v grep | head -1 || echo "  not running"
echo "=== log head ==="
head -8 runs/t3/g2_scaling.log