#!/bin/bash
# a2_run.sh — TARGET A2 (Option B): teach 24 pre-registered cluster targets, hold out 8.
#
# Settings are FROZEN by TARGET-A2.md: lam_concept=0.5, lr=0.5 taken from the cluster
# run's own frozen selection -- deliberately NOT re-tuned, so no second selection is
# introduced. K=32. 3 seeds. Validation only. Test set untouched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=32
export T3_MODELS=readable,normal
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T2_LAM_CONCEPT=0.5
export T2_CONCEPT_SET=clusters32
export T2_N_TAUGHT=24
export T3_LR=0.5
export T3_SEEDS=0,1,2
export T3_TAG=K32_a2
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a2_run.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY"
[ -f runs/t3/concept_clusters32.json ] || fail "no k=32 targets -- run a2_derive_clusters.sh"

echo "=== TARGET A2 RUN  $(date '+%F %T') ===" >> "$LOG"
echo "--- training: readable + normal, 3 seeds, lc=0.5 lr=0.5 (frozen, not re-tuned)" >> "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
RC=$?
echo "    training rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "A2 training failed rc=$RC"

echo "--- Meter-03 audit: 24 taught vs their targets, 8 held-out vs THEIR targets" >> "$LOG"
T3M_A2=1 T3M_K=32 T3M_N_SUP=24 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py \
  >> "$LOG" 2>&1
RC=$?
echo "    meter3 audit rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "A2 Meter-03 audit failed rc=$RC"

echo "=== TARGET A2 TRAINING+AUDIT DONE  $(date '+%F %T') ===" >> "$LOG"