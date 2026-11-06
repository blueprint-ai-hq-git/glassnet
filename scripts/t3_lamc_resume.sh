#!/bin/bash
# t3_lamc_resume.sh — resume step 5 AFTER the picker fix.
#
# The tuning phase already completed successfully (6/6 blocks, rc=0) and is NOT
# re-run. This script only: applies the frozen selection rule to that existing
# tuning data, then runs the 3-seed confirmations at the selected lr, then the
# audits with the circularity split.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
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
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/lamc_resume.log

fail() { echo "!!! ABORT: $*" >> "$LOG"; echo "!!! ABORT: $*"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY"

echo "=== step 5 RESUME  $(date '+%F %T') ===" >> "$LOG"

"$PY" scripts/t3_lamc_pick.py >> "$LOG" 2>&1
RC=$?
echo "    pick rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "picker failed (rc=$RC)"
[ -s runs/t3/lamc_selection.json ] || fail "selection file empty/missing"

# selection must contain all three lam_concept values
N=$(grep -c "lam_concept" runs/t3/lamc_selection.json)
[ "$N" -ge 3 ] || fail "selection has only $N entries, expected 3"
echo "    selection entries: $N" >> "$LOG"

SEL=$("$PY" -c "
import json
s=json.load(open('runs/t3/lamc_selection.json'))
for k,v in s.items(): print(v['lam_concept'], v['lr'])
")
echo "    selected: $(echo $SEL)" >> "$LOG"

while read -r LAMC LR; do
  [ -z "$LAMC" ] && continue
  echo "--- confirm lam_concept=$LAMC lr=$LR (3 seeds)  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT=$LAMC T3_LR=$LR T3_SEEDS=0,1,2 T3_TAG="K32_lc${LAMC}_lr${LR}_sel" \
    nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
  RC=$?
  echo "    confirm lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
  [ "$RC" -ne 0 ] && fail "confirmation failed for lc=$LAMC lr=$LR (rc=$RC)"
done <<< "$SEL"

export T3A_BACKEND=cupy
export T3A_LOAD_DIR="runs/t3/ckpt"
export T3A_NAMER=pa
export T3A_MIN_SUPPORT=0.01
export T3A_K=32
export T3A_SEEDS=0,1,2
export T3A_EPOCHS=96
export T3A_MIN_EPOCH=30
export T3A_PATIENCE=15
export T3A_N_SUP=16
while read -r LAMC LR; do
  [ -z "$LAMC" ] && continue
  echo "--- audit lam_concept=$LAMC lr=$LR  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT=$LAMC T3A_LR=$LR \
    nice -n 19 ionice -c3 "$PY" scripts/t3_audit.py >> runs/t3/audit.log 2>&1
  RC=$?
  echo "    audit lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
  [ "$RC" -ne 0 ] && fail "audit failed for lc=$LAMC lr=$LR (rc=$RC)"
done <<< "$SEL"

echo "=== step 5 RESUME done  $(date '+%F %T') ===" >> "$LOG"