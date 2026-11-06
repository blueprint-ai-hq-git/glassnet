#!/bin/bash
# run_f_audit.sh — do the concepts survive a generation of self-training?
# T3M_K MUST match the student's K (9). Without it the audit builds a K=32 net, loads a K=9
# checkpoint into it, and reports coverage as 9/32 -- a pure artifact (it printed 35% once).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export T3M_K=9
nice -n 19 ionice -c3 ".venv-gpu/bin/python3" -u scripts/f_audit.py 2>&1 | sed -n '/TARGET F — DO THE CONCEPTS/,$p' | tail -34