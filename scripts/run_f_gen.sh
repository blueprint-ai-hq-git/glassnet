#!/bin/bash
# run_f_gen.sh — TARGET F step 1: teacher writes the corpus, selection rules applied.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export F_N_SEEDS=4000
export F_N_NEW=32
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/f_gen_corpus.py 2>&1 | sed -n '/TARGET F — TEACHER/,$p' | tail -30