#!/bin/bash
# CPU-only wrapper for the h5l (ARM 2, --epochs 6) grader. Text only: no checkpoint is opened, no model, no GPU.
# Predictions: lens/runs/H2-H5L-PREDICTIONS-P102.txt. Output: lens/runs/H5L-ARM2-NUMBERS.txt (APPENDS a block;
# the LAST block in that file is authoritative).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
.venv-gpu/bin/python3 -u lens/scripts/h5l_arm2_numbers.py
echo "H5L-ARM2-NUMBERS EXIT=$?  $(date '+%F %H:%M:%S')"
