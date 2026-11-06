#!/bin/bash
# run_record_e.sh — log the Target E result, refresh STATUS.md, copy to /home/yahya.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_e.py 2>&1 | tail -3
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
cp TARGET-E-RESULTS.md /home/yahya/
echo "--- copied ---"
ls -la /home/yahya/GLASSNET-STATUS.md /home/yahya/TARGET-E-RESULTS.md