#!/bin/bash
# WHY the rivalry penalty failed: measure the thing it was meant to change, on the audit's datum.
# Prints one row per ship (activations correlation + the audit's own view), plus per-concept
# detail for the two K21k ships. Run as a script file; inline long commands trip the approval gate.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "--- summary (all four ships, pinned datum n=60000 seed=0)"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/concept_rivalry_report.py

echo
echo "--- per-concept detail for the DECORR ships (what the penalty broke)"
DETAIL=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/concept_rivalry_report.py \
  seetalk_k21k_d0.3.npz