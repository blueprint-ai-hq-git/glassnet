#!/usr/bin/env bash
# run_numeric_path_opt_decay.sh - the FOLLOW-UP to PLAN item (d2): does a DECAY SCHEDULE fix what a constant
# normalised step cannot do (a limit cycle)? Predictions written FIRST:
# lens/runs/NUMERIC-PATH-OPT-DECAY-PREDICTIONS.txt (P40a-i).
# Appends to lens/runs/NUMERIC-PATH-OPT-DECAY.txt (never overwrites); rows in
# lens/runs/numeric_path_decay.jsonl; the decay=1.0 regression writes lens/runs/numeric_path_opt_regress.jsonl.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet"
OUT=lens/runs/NUMERIC-PATH-OPT-DECAY.txt
export OUT_JSONL=lens/runs/numeric_path_decay.jsonl
{
  echo "===================================================================================================="
  echo "PLAN (d2) FOLLOW-UP: OUR OPTIMISER WITH A DECAY SCHEDULE - run started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "predictions: lens/runs/NUMERIC-PATH-OPT-DECAY-PREDICTIONS.txt (P40a-i, written before this run)"
  echo "the constant-step FAILs in lens/runs/NUMERIC-PATH-OPT.txt stand as measured and are not redefined"
  echo "===================================================================================================="
  python3 lens/scripts/numeric_path_opt_test.py --property
  echo
  python3 lens/scripts/numeric_path_opt_test.py --orbit
  echo
  echo "--- decay 0.999, momentum 0.0, four step sizes ---"
  OPT_KIND=relgrad OPT_DECAY=0.999 OPT_LRS=0.005,0.01,0.02,0.05 SEEDS=0,1,2 TAG=decay \
    python3 lens/scripts/numeric_path_opt_test.py
  echo
  echo "--- decay 0.999, momentum 0.9, two step sizes ---"
  OPT_KIND=relgrad OPT_DECAY=0.999 OPT_MOM=0.9 OPT_LRS=0.02,0.05 SEEDS=0,1,2 TAG=decay \
    python3 lens/scripts/numeric_path_opt_test.py
  echo
  echo "--- arm C (the harness) with decay 0.999 at lr 0.02 ---"
  ARM_ONLY=c OPT_KIND=relgrad OPT_DECAY=0.999 OPT_LRS=0.02 SEEDS=0,1,2 TAG=decay \
    python3 lens/scripts/numeric_path_opt_test.py
  echo
  python3 lens/scripts/numeric_path_opt_test.py --regress
  echo
  python3 lens/scripts/numeric_path_opt_test.py --verdict2
  echo "=== DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$OUT" 2>&1
echo "appended to $OUT"
