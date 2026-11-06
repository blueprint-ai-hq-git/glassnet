#!/bin/bash
# gpu_bench.sh — CPU vs GPU on the real T2 workload.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
LOG=runs/gpu_bench.log
VENV="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu"
echo "=== real-workload CPU vs GPU bench  $(date '+%F %T') ===" > "$LOG"
nice -n 10 "$VENV/bin/python" scripts/gpu_bench.py >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
cat "$LOG"