#!/bin/bash
# dead_run_diag.sh — reproduce the dead readable run and inspect its internals.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
LOG=runs/t2/dead_diag.log
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
echo "=== dead-run diagnostic  $(date '+%F %T') ===" > "$LOG"
nice -n 19 ionice -c3 "$PY" scripts/dead_run_diag.py >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
cat "$LOG"