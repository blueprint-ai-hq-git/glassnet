#!/usr/bin/env bash
# correct_numeric_path_opt.sh - the CORRECTION BLOCK for the (d2) follow-up: re-measures and prints the two
# bars of mine that FAILED (P40a's mechanism, P40h's single-step value) with the compared values beside the
# verdict. No bar is moved; both stand FAILED as worded. Appends, never overwrites.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
export OUT_JSONL=lens/runs/numeric_path_decay.jsonl
OUT=lens/runs/NUMERIC-PATH-OPT-DECAY.txt
{
  echo ""
  echo "===================================================================================================="
  echo "CORRECTION BLOCK $(date '+%Y-%m-%d %H:%M:%S'): the two bars of mine that FAILED, re-measured and"
  echo "printed with the compared values beside the verdict. No bar was moved; both stand FAILED as worded."
  echo "===================================================================================================="
  python3 lens/scripts/numeric_path_opt_test.py --property
  echo
  python3 lens/scripts/numeric_path_opt_test.py --orbit
  echo
  python3 lens/scripts/numeric_path_opt_test.py --verdict2
  echo "=== CORRECTION DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
