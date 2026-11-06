#!/bin/bash
# run_diag_gpu.sh — minimal GPU launch repro, unbuffered so a crash shows where it died.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
echo "exit code will be shown below; 139 = segfault"
.venv-gpu/bin/python3 -u lens/tests/diag_gpu_launch.py 2>&1 | tail -26
echo "rc=${PIPESTATUS[0]}"