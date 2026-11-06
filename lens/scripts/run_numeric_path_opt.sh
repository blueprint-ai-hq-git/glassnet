#!/usr/bin/env bash
# run_numeric_path_opt.sh - PLAN item (d2): the numeric path under OUR OWN optimiser (lens/opt.py).
# Predictions written FIRST: lens/runs/NUMERIC-PATH-OPT-PREDICTIONS.txt (P39a-j).
# Appends to lens/runs/NUMERIC-PATH-OPT.txt (never overwrites); rows also land in
# lens/runs/numeric_path_opt.jsonl, so an interrupted run keeps everything it measured.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
OUT=lens/runs/NUMERIC-PATH-OPT.txt
{
  echo "===================================================================================================="
  echo "PLAN (d2) THE NUMERIC PATH WITH OUR OWN SCALE-INVARIANT OPTIMISER - run started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "predictions: lens/runs/NUMERIC-PATH-OPT-PREDICTIONS.txt (P39a-j, written before this run)"
  echo "harness: lens/scripts/numeric_path_test.py IMPORTED unchanged; only the step rule differs"
  echo "===================================================================================================="
  python3 lens/scripts/numeric_path_opt_test.py --property
  echo
  OPT_KIND=sgd     OPT_LRS=0.05,0.1,0.2,0.5   SEEDS=0,1,2 TAG=sweep python3 lens/scripts/numeric_path_opt_test.py
  echo
  OPT_KIND=relgrad OPT_LRS=0.005,0.01,0.02,0.05 SEEDS=0,1,2 TAG=sweep python3 lens/scripts/numeric_path_opt_test.py
  echo
  OPT_KIND=relgrad OPT_MOM=0.9 OPT_LRS=0.02,0.05 SEEDS=0,1,2 TAG=sweep python3 lens/scripts/numeric_path_opt_test.py
  echo
  ARM_ONLY=c OPT_KIND=relgrad OPT_LRS=0.02 SEEDS=0,1,2 TAG=sweep python3 lens/scripts/numeric_path_opt_test.py
  echo
  python3 lens/scripts/numeric_path_opt_test.py --verdict
  echo "=== DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
