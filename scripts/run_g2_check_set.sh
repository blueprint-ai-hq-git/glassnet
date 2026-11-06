#!/bin/bash
# run_g2_check_set.sh — confirm which concept set the G2 sweep trained on.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
env -u T2_CONCEPT_SET -u T2_N_TAUGHT nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  scripts/g2_check_concept_set.py 2>&1 | sed -n '/WHICH CONCEPT SET/,$p' | tail -18