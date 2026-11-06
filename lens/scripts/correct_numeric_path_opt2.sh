#!/usr/bin/env bash
# correct_numeric_path_opt2.sh - FINAL VERDICT BLOCK for the (d2) follow-up. The two script defects fixed
# here are mine, not the experiment's: the dedupe key collapsed the property rows (they share a seed), and
# the corrected-property comparison used an ABSOLUTE tolerance on a ratio of up to 32. No bar was moved.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
export OUT_JSONL=lens/runs/numeric_path_decay.jsonl
OUT=lens/runs/NUMERIC-PATH-OPT-DECAY.txt
{
  echo ""
  echo "===================================================================================================="
  echo "FINAL VERDICT BLOCK $(date '+%Y-%m-%d %H:%M:%S') - two DEFECTS IN MY OWN CHECK SCRIPT FIXED, no bar moved:"
  echo "  (1) the dedupe key collapsed the property rows (they all share seed -1), so the rmsprop rows were"
  echo "      invisible to the verdict - now keyed on mode/scale/n as well;"
  echo "  (2) the corrected-property comparison used an absolute 1e-6 on a ratio that reaches 32 (float32"
  echo "      tensors), so it read MISMATCH on correct numbers - now relative, 1e-5."
  echo "===================================================================================================="
  python3 lens/scripts/numeric_path_opt_test.py --verdict2
  echo "=== FINAL VERDICT DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
