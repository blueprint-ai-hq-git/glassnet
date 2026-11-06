#!/bin/bash
# run_agree_numbers.sh -- append a fresh GRADE block for the AGREEMENT arm (agree_sgd04_s2) to
# lens/runs/GRADE-P61.txt. APPEND-ONLY: the grader never rewrites an earlier block, so the ep1+ep2 grade of
# 2026-10-26 18:2x stays on the record when a later wake grades more epochs.
#
# READ-ONLY over the arm: no checkpoint is loaded, no model is built, no GPU, no generation. The arm keeps
# training. CPU only, a few seconds.
#
# Predictions for the grader: lens/runs/AGREE-GRADE-PREDICTIONS-P113.txt (written before it existed).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/GRADE-P61.txt
{
  echo ""
  echo "####################################################################################################"
  echo "# GRADE BLOCK appended $(date '+%F %H:%M:%S') by lens/scripts/run_agree_numbers.sh"
  echo "# grader: lens/scripts/agree_numbers.py   predictions: lens/runs/AGREE-GRADE-PREDICTIONS-P113.txt"
  echo "####################################################################################################"
  nice -n 19 python3 lens/scripts/agree_numbers.py
  echo "# grader exit=$?"
} >> "$OUT" 2>&1
tail -20 "$OUT"
