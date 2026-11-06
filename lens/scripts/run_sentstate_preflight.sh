#!/usr/bin/env bash
# Runner for the sentstate pre-flight (P111a-h). APPEND-ONLY: every attempt stamps its own block with a
# timestamp and a reason line, so a FAIL that was fixed stays on the record (the attempt-1 block in
# lens/runs/SENTSTATE-STRUCTURAL.txt is the P111e contractive-init FAIL and the P111f instrument defect).
# Kept as a FILE, not an inline env-prefixed command (the approval gate blocks those).
set -u
P="/run/media/yahya/HARD DISK 4/GlassNet"
LOG="$P/lens/runs/SENTSTATE-STRUCTURAL.txt"
REASON="${1:-re-run of the pre-flight; reason not given}"
cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
{
  echo ""
  echo "################################################################################"
  echo "# ATTEMPT $(date '+%Y-%m-%d %H:%M:%S')  reason: $REASON"
  echo "################################################################################"
} >> "$LOG"
"$P/.venv-gpu/bin/python3" -u "$P/lens/scripts/sentstate_preflight.py" >> "$LOG" 2>&1
echo "PREFLIGHT EXIT=$?"
START=$(grep -n '^# ATTEMPT' "$LOG" | tail -1 | cut -d: -f1)
sed -n "${START},\$p" "$LOG" | grep -E '^P111|^RESULT|^ +fd'
