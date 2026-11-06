#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/chat_data.py 2>&1 | tail -25