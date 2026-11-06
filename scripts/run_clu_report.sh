#!/bin/bash
# run_clu_report.sh — the cluster results report.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
"$PY" scripts/t3_clu_report.py 2>&1