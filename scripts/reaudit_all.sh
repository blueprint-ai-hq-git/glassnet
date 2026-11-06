#!/bin/bash
# reaudit_all.sh — Meter-02 re-audit of everything, resumable.
# Unbuffered so a silent kill still leaves a readable log.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/reaudit_all.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }

echo "=== METER-02 RE-AUDIT  $(date '+%F %T') ===" >> "$LOG"

echo "--- T2 concept audits" >> "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_reaudit_t2.py >> "$LOG" 2>&1
RC=$?
echo "    t2 rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "T2 re-audit failed rc=$RC"

echo "--- rungs 1/2/3 (fidelity meter + floor)" >> "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_reaudit_rungs.py >> "$LOG" 2>&1
RC=$?
echo "    rungs rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "rung re-audit failed rc=$RC"

echo "=== METER-02 RE-AUDIT DONE  $(date '+%F %T') ===" >> "$LOG"