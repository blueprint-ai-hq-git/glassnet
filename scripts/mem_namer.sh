#!/bin/bash
# mem_namer.sh — each mode in a fresh process.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
echo "############ FULL-SET FORWARD (current namer behaviour) ############"
"$PY" scripts/mem_namer.py full 2>&1 | tail -4
echo
echo "############ CHUNKED FORWARD (20k windows at a time) ############"
"$PY" scripts/mem_namer.py chunked 2>&1 | tail -4