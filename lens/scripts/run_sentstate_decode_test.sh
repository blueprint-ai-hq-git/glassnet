#!/bin/bash
# run_sentstate_decode_test.sh -- P115a-f, CPU only, 2 cores at nice 19 (the agreement arm holds the GPU and
# the machine is at its RAM ceiling; this must not compete).
# Predictions: lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt (written before the code it tests).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/sentstate_decode_test.py
echo "DECODE TEST EXIT=$?  $(date '+%F %H:%M:%S')"
