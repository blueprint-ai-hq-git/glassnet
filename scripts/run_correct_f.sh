#!/bin/bash
# run_correct_f.sh — correct the Target F record.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/correct_f.py 2>&1 | tail -28
cp STATUS.md TARGET-F-RESULTS.md /home/yahya/
echo "--- copied ---"