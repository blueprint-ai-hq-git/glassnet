#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# PREDICTIONS written BEFORE running (shipping config, K=22, big rung):
#  p1 gates 4/4
#  p2 val accuracy 77-80% and delta vs control > +14.51 (the K12 big number): K22 bought +4.4 at
#     small scale; if the delta does NOT grow at big scale the K-gain was small-data-specific
#  p3 checkpoint round-trip bit-identical (0.000e+00)
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 14 --n 60000 --device gpu &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
