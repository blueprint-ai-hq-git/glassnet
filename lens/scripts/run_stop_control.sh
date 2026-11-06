#!/bin/bash
# Does the model know when to stop? CPU only, no training, existing checkpoint.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
CHAT_DATA_DIR="$PWD/data/chat/prepared_ext" \
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/stop_control.py