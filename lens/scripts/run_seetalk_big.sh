#!/bin/bash
# see+talk at the BIG rung: full Fashion-MNIST train (60k images), d=128, GPU.
# Pre-registered predictions, written BEFORE the run:
#   p1: val accuracy lands in 80-92% (small/medium were 77.3/77.5 with 10k/20k images;
#       medium was flat vs small, so growth from data alone must be modest — 15x data should
#       push past the template-word ceiling toward the class-word share)
#   p2: the text-only control stays at ~61-62% (its ceiling is the template words it can guess
#       blind; more data cannot beat that) -> delta >= +18
#   p3: g1 shows >=6 distinct nouns from 10 images (medium showed 5, small 6)
# If p2 fails (control rises), the control was capacity-limited, not information-limited, and
# the delta claim must be restated — that is a finding, not a failure to hide.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk.py \
  --d 128 --epochs 14 --n 60000 --device gpu --out seetalk_big &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
