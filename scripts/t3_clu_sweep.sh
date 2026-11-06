#!/bin/bash
# t3_clu_sweep.sh — condition 4: WORD-CLUSTER concept targets (v2).
#
# ONE change vs the single-word run: the concept TARGETS are clusters
# (CONCEPT-TARGETS-02.md, method pre-registered). Everything else at baseline:
# K=32, lam_act 0.2, lam_w 1.0, position-aware namer, support 1%, Meter-02 audit.
#
# Budget identical to v1: lam_concept x lr for tuning (1 seed, frozen
# SELECTION-RULE.md), then 3 seeds at the selected lr, both models, validation only.
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
export T2_CONCEPT_SET=clusters
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/clu_sweep.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY"
[ -f runs/t3/concept_clusters.json ] || fail "no cluster targets — run t3_concept_clusters.py first"

echo "=== v2 WORD-CLUSTER SWEEP  $(date '+%F %T') ===" >> "$LOG"

# ---- phase 1: tune (1 seed, both models) ----
for LAMC in 0.1 0.5 1.0; do
  for LR in 0.5 1.0; do
    echo "--- tune clusters lam_concept=$LAMC lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_CONCEPT=$LAMC T3_LR=$LR T3_SEEDS=0 T3_TAG="K32_clu${LAMC}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
    RC=$?
    echo "    tune lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
    [ "$RC" -ne 0 ] && fail "tuning failed lc=$LAMC lr=$LR rc=$RC"
  done
done

ls runs/t3/models_K32_clu*_tune.json >/dev/null 2>&1 || fail "no cluster tuning files"

echo "--- picking lr per lam_concept  $(date '+%T')" >> "$LOG"
"$PY" scripts/t3_clu_pick.py >> "$LOG" 2>&1
RC=$?
echo "    pick rc=$RC" >> "$LOG"
[ "$RC" -ne 0 ] && fail "picker failed rc=$RC"
[ -s runs/t3/clu_selection.json ] || fail "selection file empty"

SEL=$("$PY" -c "
import json
s=json.load(open('runs/t3/clu_selection.json'))
for k,v in s.items(): print(v['lam_concept'], v['lr'])
")
echo "    selected: $(echo $SEL)" >> "$LOG"

# ---- phase 2: 3 seeds at the selected lr ----
while read -r LAMC LR; do
  [ -z "$LAMC" ] && continue
  echo "--- confirm clusters lam_concept=$LAMC lr=$LR (3 seeds)  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT=$LAMC T3_LR=$LR T3_SEEDS=0,1,2 T3_TAG="K32_clu${LAMC}_lr${LR}_sel" \
    nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
  RC=$?
  echo "    confirm lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
  [ "$RC" -ne 0 ] && fail "confirmation failed lc=$LAMC lr=$LR rc=$RC"
done <<< "$SEL"

# ---- phase 3: Meter-02 audit (2% firing floor), supervised/unsupervised split ----
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
export T3A_FIRING_FLOOR=0.02
export T3A_LIVE_TARGET=0.02
while read -r LAMC LR; do
  [ -z "$LAMC" ] && continue
  echo "--- Meter-02 audit clusters lam_concept=$LAMC lr=$LR  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT=$LAMC T3A_LR=$LR \
    nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py >> runs/t3/audit.log 2>&1
  RC=$?
  echo "    audit lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
  [ "$RC" -ne 0 ] && fail "audit failed lc=$LAMC lr=$LR rc=$RC"
done <<< "$SEL"

echo "=== v2 WORD-CLUSTER SWEEP DONE  $(date '+%F %T') ===" >> "$LOG"