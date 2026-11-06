#!/bin/bash
# run_diag_cupy.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u lens/tests/diag_cupy.py 2>&1 | tail -30