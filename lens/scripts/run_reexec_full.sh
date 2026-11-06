#!/bin/bash
# run_reexec_full.sh — re-run the code execution test and keep the WHOLE output.
#
# Why: the stored code1_reexec.json only kept the last write (the AFTER-join-fix case) and the
# stdout was lost, so the concept-match check and the BEFORE/AFTER comparison were never seen.
# The number that matters is whether the checkpoint loads with the concepts it was trained on.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
CKPT="${CKPT:-code1_ep16}" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tee /home/yahya/reexec_full.txt | tail -70
