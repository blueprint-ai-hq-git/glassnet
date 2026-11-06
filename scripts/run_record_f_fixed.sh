#!/bin/bash
# run_record_f_fixed.sh — record the corrected F concept result.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_f_fixed.py 2>&1 | tail -3
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
echo "=== the LM numbers from the fixed run (the collapse, for the record) ==="
grep -E "STUDENT|ep  40" runs/t3/f_fixed.log | tail -3
echo "=== jobs running (should be none) ==="
ps -eo pid,cmd | grep -E "f_train|f_audit|t3_models" | grep -v grep || echo "  none"