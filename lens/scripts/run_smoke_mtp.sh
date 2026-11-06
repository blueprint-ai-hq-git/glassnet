#!/bin/bash
# Run the MTP smoke gate (CPU, tiny, a couple of minutes).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/smoke_mtp.py
echo "exit=$?"