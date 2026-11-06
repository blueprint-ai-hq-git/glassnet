#!/bin/bash
# CHAIN 5 — after the merged-192 run finishes, run the four-option comparison matrix.
# Bracket pattern in pgrep so this script never matches its own command line.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
while pgrep -f "[t]rain_chat.py .*--out t12_merged_d512" >/dev/null 2>&1; do sleep 60; done
echo "merged-192 finished at $(date +%H:%M) -- starting the comparison matrix"
bash lens/scripts/run_matrix.sh > /tmp/matrix.log 2>&1
echo "matrix finished at $(date +%H:%M)"
.venv-gpu/bin/python3 -u lens/scripts/matrix_summary.py > /tmp/matrix_summary.log 2>&1
tail -30 /tmp/matrix_summary.log