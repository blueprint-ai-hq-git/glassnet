#!/bin/bash
# run_record_g.sh — record Target G's result + the G2 pre-registration.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_g.py 2>&1 | tail -3
cp STATUS.md TARGET-G.md TARGET-G-RESULTS.md TARGET-G2.md /home/yahya/
echo "=== copied to /home/yahya ==="
echo "=== G2 progress so far ==="
grep -E "^  D=" runs/t3/g2_scaling.log | tail -8 || echo "  (first run still starting)"