#!/bin/bash
# soft-label teacher experiment. PREDICTIONS, written BEFORE running:
#  P1 gradcheck distill_kl passes at BOTH tau=1 and tau=2 (else nothing else runs)
#  P2 B2 (soft teacher) beats A (10% real, no teacher) by >2 pts: the teacher distribution carries
#     probability on the right noun even when its argmax is wrong — the information argmax lacks
#  P3 B2 stays within 10 pts of C (100% real)
# If P2 fails again the conclusion is that a SAME-ARCHITECTURE teacher has no spare information at
# any label softness, and the next experiment must be a STRONGER teacher (bigger d, more data)
# labelling the hidden 90% — stated up front so a second t2 failure is a mapped path, not a surprise.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/teacher_soft.py \
  --epochs 8 --d 48 --frac 0.1 --tau 2.0 --seeds 0,1,2 --device cpu 2>&1 | tail -40
