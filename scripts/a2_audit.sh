#!/bin/bash
# a2_audit.sh — the Meter-03 audit for A2 only (training already completed rc=0).
# Re-run after fixing KIND: the first attempt loaded the WORDS targets (16) while
# N_SUP=24, which raised IndexError on the off-diagonal loop.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a2_run.log

echo "--- Meter-03 audit (retry, KIND fixed)  $(date '+%T')" >> "$LOG"
T3M_A2=1 T3M_K=32 T3M_N_SUP=24 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py \
  >> "$LOG" 2>&1
RC=$?
echo "    meter3 audit rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && { echo "!!! ABORT: audit rc=$RC" | tee -a "$LOG"; exit 1; }
echo "=== A2 AUDIT DONE  $(date '+%F %T') ===" >> "$LOG"