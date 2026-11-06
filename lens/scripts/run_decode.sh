#!/bin/bash
# run_decode.sh — write the trained model out in words, and print its number.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_to_words.py "$@" 2>&1 | tail -60