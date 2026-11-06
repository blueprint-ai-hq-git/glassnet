#!/usr/bin/env bash
# final_block_opt.sh - the last evidence block for PLAN item (d2): the shipped callers smoke test and the
# shipped (d) harness re-run after the lens/opt.py change. Appends, never overwrites.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
OUT=lens/runs/NUMERIC-PATH-OPT-DECAY.txt
{
  echo ""
  echo "===================================================================================================="
  echo "SHIPPED-PATH CHECKS $(date '+%Y-%m-%d %H:%M:%S'): the callers of lens/opt.py, and the stored (d) harness"
  echo "===================================================================================================="
  python3 lens/scripts/opt_smoke.py
  echo
  python3 lens/scripts/regress_after_opt_change.py
  echo "=== CHECKS DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
