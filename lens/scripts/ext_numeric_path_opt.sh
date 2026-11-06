#!/usr/bin/env bash
# ext_numeric_path_opt.sh - the labelled EXTENSION to the (d2) follow-up: the same decay 0.999 over a 100x
# range of the step size (lr 0.1, 0.2, 0.5). Prediction written first in
# lens/runs/NUMERIC-PATH-OPT-EXT-PREDICTIONS.txt (P41). Appends, never overwrites.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
export OUT_JSONL=lens/runs/numeric_path_decay.jsonl
OUT=lens/runs/NUMERIC-PATH-OPT-DECAY.txt
{
  echo ""
  echo "===================================================================================================="
  echo "EXTENSION $(date '+%Y-%m-%d %H:%M:%S'): the same decay over a 100x step-size range (P41, declared"
  echo "before this run in lens/runs/NUMERIC-PATH-OPT-EXT-PREDICTIONS.txt). Not part of P40's grid."
  echo "===================================================================================================="
  OPT_KIND=relgrad OPT_DECAY=0.999 OPT_LRS=0.1,0.2,0.5 SEEDS=0,1,2 TAG=ext \
    python3 lens/scripts/numeric_path_opt_test.py
  echo
  echo "--- P41 VERDICT (bar: unseen >= 90% and 3/3 finite at EACH of 0.1, 0.2, 0.5) ---"
  python3 lens/scripts/numeric_path_opt_ext_verdict.py
  echo "=== EXTENSION DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
