#!/bin/bash
# The warm concept memory tier at CORPUS SCALE: does memory stay the store's own bytes only?
# PLAN.txt item (c) RUN. CPU only, no GPU, about 20 minutes for 294M tokens. Output: WARM-MEMORY-STREAM.txt
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_honest"
export MILESTONES="${MILESTONES:-1000000,10000000,50000000,100000000}"
PY=.venv-gpu/bin/python3
[ -x "$PY" ] || PY=python3
nice -n 19 ionice -c3 "$PY" -u lens/scripts/warm_memory_stream.py > lens/runs/WARM-MEMORY-STREAM.txt 2>&1
echo "exit=$?" >> lens/runs/WARM-MEMORY-STREAM.txt
