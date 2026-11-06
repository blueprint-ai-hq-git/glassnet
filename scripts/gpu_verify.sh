#!/bin/bash
# gpu_verify.sh — launch the GPU verification in the isolated GPU venv.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs
LOG=runs/gpu_verify.log
VENV="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu"
{
echo "=== GPU verification  $(date '+%F %T') ==="
} > "$LOG"
nice -n 10 "$VENV/bin/python" scripts/gpu_verify.py >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
tail -60 "$LOG"