#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/eyes56.py \
  --n-per 600 --epochs 12 --seeds 0,1,2 --device gpu &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
