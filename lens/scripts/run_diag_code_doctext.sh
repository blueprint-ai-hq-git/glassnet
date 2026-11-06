#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
CODE_DOC="${CODE_DOC:-code}" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/diag_code_doctext.py 2>&1 | tail -45