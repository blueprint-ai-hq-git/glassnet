#!/bin/bash
# THE DECISIVE DISTILLATION TEST. Teacher = d128 trained on images 10000-29999; student rows come
# from images 0-9999. ZERO overlap: the teacher cannot have memorised the student's answer key.
# If soft labels from THIS teacher still fail t2, the honest conclusion is that on this task the
# teacher's extra information over 10% real labels is essentially nil — a mapped negative, not a
# retry loop.
# PREDICTIONS (before running):
#  P1 vocab order: teacher/student vocab ORDER may differ -> remap note prints (integrity check)
#  P2 teacher top1 mean on student rows ~0.6-0.85 (it is 78% accurate, confident, disjoint)
#  P3 B2 > A by >2 pts AND outside noise  [t2] — this is what distillation is supposed to do; if
#     it fails here the negative result is FINAL at this rung and gets written as one
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/teacher_soft.py \
  --teacher "lens/runs/seetalk/seetalk_teacher_d128.npz" \
  --epochs 8 --d 48 --frac 0.1 --tau 2.0 --seeds 0,1,2 --device gpu 2>&1 | tail -26
