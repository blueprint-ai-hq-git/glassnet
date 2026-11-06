#!/bin/bash
# SELF-CONTEXT ARM — THE NUMBERS AND THEIR VERDICTS (P99a-e). Text only: no checkpoint, no model, no GPU, no
# trainer. Predictions lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt, written BEFORE selfctx_numbers.py existed.
# Output: lens/runs/SELFCTX-NUMBERS.txt (new file; nothing stored is written to).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/SELFCTX-NUMBERS.txt
{
  echo "===================================================================================================="
  echo "SELF-CONTEXT ARM — NUMBERS RUN  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt (written before this runner existed)"
  echo "===================================================================================================="
  .venv-gpu/bin/python3 -u lens/scripts/selfctx_numbers.py
  echo "NUMBERS EXIT=$?  $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
tail -c 200 "$OUT"
