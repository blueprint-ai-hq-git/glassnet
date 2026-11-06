#!/bin/bash
# concept ceiling test. PRE-REGISTERED predictions (written before running):
#  Q1 the DUMMY arm does NOT beat base (c3 holds): random projections are not understanding
#  Q2 the REAL arm beats base by >2: more real named concepts break the ~77% band
#  Q3 real > dummy by >2 (c1): the gain is the CONCEPTS, not the extra input width
# If Q2 fails but c3 holds: the 12 concepts already cover what this image size can express and
# the ceiling is PIXELS or TASK, not the concept count — also a real answer.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/concepts_more.py \
  --n 10000 --d 64 --epochs 12 --seeds 0,1,2 --device gpu &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
