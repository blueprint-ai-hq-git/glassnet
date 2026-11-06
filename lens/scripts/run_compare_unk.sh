#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code
export CD_VOCAB=6000
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/compare_unk_rate.py 2>&1 | tail -30
