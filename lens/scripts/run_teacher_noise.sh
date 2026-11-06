#!/bin/bash
# THE CUSTOMER SCENARIO: 30% of the 10% real labels corrupted (same key for all seeds/arms;
# VAL stays clean). The soft arm never reads real labels, so teacher output is corruption-free.
# PREDICTIONS written BEFORE running (the measured boundary from the clean-label negative says
# the teacher CAN only win where labels are worse than the teacher's estimate):
#  P1  arm A (noisy 10% labels) drops ~15-25 pts below the clean 66.5 (30% random targets poison)
#  P2  arm B2 (teacher, never sees the corruption) stays near its clean soft number (~66)
#  P3  t2 FINALLY passes: B2 - A > 2 and outside noise. THIS is the condition under which the
#      teacher earns its place; if t2 STILL fails at 30% noise, the whole distillation line is
#      closed on evidence at this task, and PLAN step-5 says so without retrying further.
#  P4  arm C (100% noisy) lands BETWEEN A and the clean C (~60-72): more bad data is still bad.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/teacher_soft.py \
  --teacher "lens/runs/seetalk/seetalk_teacher_d128.npz" \
  --epochs 8 --d 48 --frac 0.1 --tau 2.0 --noise 0.3 --seeds 0,1,2 --device gpu 2>&1 | tail -14
