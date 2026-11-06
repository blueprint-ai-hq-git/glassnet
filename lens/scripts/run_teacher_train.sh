#!/bin/bash
# STRONG, DISJOINT teacher: d=128, 20k images STARTING AT 10000. The student pool is images
# 0..9999, so this teacher has never seen a single row the student will train on. If the first
# soft-teacher t2 failure (-0.64) meant "same-pool teacher = regurgitating answer key", this one
# can only be right by GENERALISING, which is the information distillation is supposed to transfer.
# PREDICTIONS written before running:
#  P1 teacher val acc lands 72-80% (the in-pool big run made 75.51; disjoint pool should be close —
#     same task, same size, and 20k images is between small and medium rungs)
#  P2 teacher's top1 confidence on its own predictions ~0.6-0.85; entropy clearly below uniform
#  P3 its SOFT labels on the disjoint student pool keep >15% probability on the TRUE word even when
#     argmax is wrong (that spread IS the distillation signal; if it's absent, mechanism (b) cannot
#     work on this task either, and the negative result stands)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk.py \
  --d 128 --epochs 14 --n 20000 --img-start 10000 --device gpu \
  --out seetalk_teacher_d128 &
PID=$!
bash lens/scripts/game_guard.sh "$PID" 2>/dev/null &
GPID=$!
wait $PID
kill $GPID 2>/dev/null
