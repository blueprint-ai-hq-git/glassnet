#!/bin/bash
# teacher loop, tiny rung. Pre-registered predictions (written BEFORE running):
#   P1  arm B (10% data + teacher labels) lands within 10 points of arm C (100% real)   [gate t1]
#   P2  arm B beats arm A (10% real, no teacher) by >2 points AND outside seed noise    [gate t2]
#   P3  with --teacher-on off, arms A and B are the same data path and must score ~equal
#       (the switch is real: off = no teacher benefit, on = teacher benefit; if "off" still
#       wins, the win is coming from something other than the teacher)
# Failed prediction = finding, stated as such. 3 seeds, cpu, tiny.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/teacher_loop.py \
  --epochs 8 --d 48 --frac 0.1 --seeds 0,1,2 --teacher-on on --device cpu 2>&1 | tail -30
