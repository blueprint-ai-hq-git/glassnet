#!/bin/bash
# c_audit.sh — TARGET-C audit: Meter-03 on the 6 taught meaningful concepts, 3 seeds.
# Bars unchanged (AUROC 0.55, contribution 0.25, firing floor 2%). Test set untouched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_C=1
export T3M_K=16
export T3M_N_SUP=6
export T3M_IMPORT_ONLY=0
PY=".venv-gpu/bin/python3"
OUT=runs/t3/c_audit.log

echo "=== TARGET C AUDIT (Meter-03 on 6 meaningful concepts)  $(date '+%F %T') ===" | tee "$OUT"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py > "$OUT.body" 2>&1
RC=$?
cat "$OUT.body" >> "$OUT"
echo "audit rc=$RC" | tee -a "$OUT"
[ "$RC" -ne 0 ] && { echo "!!! AUDIT FAILED — see $OUT.body"; tail -25 "$OUT.body"; exit 1; }
grep -E "TAUGHT|MEAN:|PASS|pass" "$OUT.body" | tail -30