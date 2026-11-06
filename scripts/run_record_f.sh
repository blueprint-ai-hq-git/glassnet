#!/bin/bash
# run_record_f.sh — log Target F, refresh STATUS.md, copy the docs to /home/yahya.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_f.py 2>&1 | tail -4
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
cp TARGET-F-RESULTS.md LIT-CHECK-SELFIMPROVE.md /home/yahya/
echo "--- copied ---"
ls -la /home/yahya/GLASSNET-STATUS.md /home/yahya/TARGET-F-RESULTS.md /home/yahya/LIT-CHECK-SELFIMPROVE.md