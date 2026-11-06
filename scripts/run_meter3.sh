#!/bin/bash
# run_meter3.sh — Meter-03 audit on existing checkpoints (no retraining).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/meter3.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
echo "=== METER-03  $(date '+%F %T') ===" >> "$LOG"

echo "--- single words (v1 checkpoints)" >> "$LOG"
T3M_CKS=0 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py >> "$LOG" 2>&1
RC=$?
echo "    words rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "words audit failed rc=$RC"

echo "--- clusters (v2 checkpoints)" >> "$LOG"
T3M_CKS=1 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py >> "$LOG" 2>&1
RC=$?
echo "    clusters rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "clusters audit failed rc=$RC"

echo "=== METER-03 DONE  $(date '+%F %T') ===" >> "$LOG"