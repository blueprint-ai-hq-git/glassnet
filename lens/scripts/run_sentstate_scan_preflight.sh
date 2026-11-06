#!/usr/bin/env bash
# Runner for the sentstate SCAN pre-flight (P112a-f). APPEND-ONLY log; every attempt is stamped.
set -u
P="/run/media/yahya/HARD DISK 4/GlassNet"
LOG="$P/lens/runs/SENTSTATE-SCAN-STRUCTURAL.txt"
REASON="${1:-re-run of the scan pre-flight; reason not given}"
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
{
  echo ""
  echo "################################################################################"
  echo "# ATTEMPT $(date '+%Y-%m-%d %H:%M:%S')  reason: $REASON"
  echo "################################################################################"
} >> "$LOG"
"$P/.venv-gpu/bin/python3" -u "$P/lens/scripts/sentstate_scan_preflight.py" >> "$LOG" 2>&1
echo "SCAN-PREFLIGHT EXIT=$?"
START=$(grep -n '^# ATTEMPT' "$LOG" | tail -1 | cut -d: -f1)
sed -n "${START},\$p" "$LOG" | grep -E '^P112|^RESULT|^ +fd'
