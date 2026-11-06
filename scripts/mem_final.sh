#!/bin/bash
# mem_final.sh — peak RSS of the REAL namer path, before/after the fix.
# "namer" mode uses the fixed code path (_fired_matrix: chunked + gc).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "############ REAL namer path (fixed) ############"
"$PY" scripts/mem_namer.py namer 2>&1 | tail -4
echo
echo "############ old full-set forward (what it used to do) ############"
"$PY" scripts/mem_namer.py full 2>&1 | tail -3