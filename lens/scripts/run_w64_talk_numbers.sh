#!/bin/bash
# run_w64_talk_numbers.sh — print the W64 talk numbers beside their source lines (text only, CPU, seconds).
# Predictions for this file and the repair it belongs to: lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -m py_compile lens/scripts/w64_talk_numbers.py || exit 2
echo "py_compile OK"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/w64_talk_numbers.py
echo "w64_talk_numbers.py exit=$?"
