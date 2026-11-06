#!/bin/bash
# run_g2_verify.sh — control first: the G2 audit must reproduce Target E's known 7/9.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u scripts/g2_audit_verify.py 2>&1 \
  | sed -n '/CONTROL — does the G2 audit/,$p' | tail -24