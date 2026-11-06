#!/bin/bash
# resume_meter3_clusters.sh — finish the clusters half of the Meter-03 audit.
# The words half already completed (runs/t3/meter3_words.json, written 17:50) before
# the power cut; only the clusters run died partway. Re-scores existing checkpoints
# only -- no retraining.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/meter3.log

echo "=== METER-03 CLUSTERS RESUMED after reboot  $(date '+%F %T') ===" >> "$LOG"
T3M_CKS=1 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py >> "$LOG" 2>&1
RC=$?
echo "    clusters rc=$RC" >> "$LOG"
if [ "$RC" -ne 0 ]; then echo "!!! ABORT: clusters audit rc=$RC" | tee -a "$LOG"; exit 1; fi
echo "=== METER-03 COMPLETE  $(date '+%F %T') ===" >> "$LOG"