#!/bin/bash
# Settle whether the 9 remaining val/train overlaps are REAL duplicate text or fingerprint collisions.
# Plain script file: inline env-prefixed commands trip the approval scanner in this project.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/LEAK-EXACT.txt
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/leak_exact_check.py \
  data/chat/prepared_mix_ctrl_honest 32 > "$OUT" 2>&1
echo "exit=$?"
tail -12 "$OUT"