#!/bin/bash
# run_record_regressions.sh — record the regression suite.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_regressions.py 2>&1 | tail -3
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
echo "=== G2 progress ==="
.venv-gpu/bin/python3 -c "
import json,os
p='runs/t3/g2_scaling.json'
d=json.load(open(p)) if os.path.exists(p) else []
print(f'  {len(d)} runs in JSON (smoke writes first; full run overwrites)')
"
ps -eo pid,etime,cmd | grep g2_scaling | grep -v grep | head -1 || echo "  G2 finished"