#!/bin/bash
# verify_namer_fix.sh — the namer memory fix MUST NOT change any number.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "############ R1/R2 regression: floor=0 must reproduce recorded Meter-01 ############"
"$PY" scripts/t3_meter02_check.py 2>&1 | tail -10
echo
echo "############ memory: chunked forward with the fix in place ############"
"$PY" scripts/mem_namer.py chunked 2>&1 | tail -3