#!/bin/bash
# t3_lamc_sweep.sh — step 5: the owner-approved auxiliary concept loss.
#
# ONE change: lam_concept in {0.1, 0.5, 1.0}. Everything else at baseline
# (K=32, lam_act 0.2, lam_w 1.0, position-aware namer, support 1%).
# Budget identical for both models: lr in {0.5,1.0} x 1 seed (frozen joint rule),
# then 3 seeds at the selected lr, then audit with the circularity split.
#
# lam_concept=0 is the labeled UNSUPERVISED BASELINE (already measured).
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
# GPU venv (Python 3.11 + cupy). NOT the Hermes venv -- that one has no cupy and
# the whole sweep dies instantly with ModuleNotFoundError.
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/lamc_sweep.log

# Fail loudly: a silent failure that still prints "done" is worse than a crash.
fail() { echo "!!! ABORT: $*" >> "$LOG"; echo "!!! ABORT: $*"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing from $PY — wrong interpreter"

echo "=== step 5 auxiliary concept loss  $(date '+%F %T') ===" >> "$LOG"

# ---- phase 1: tune lr per lam_concept (1 seed, both models) ----
for LAMC in 0.1 0.5 1.0; do
  for LR in 0.5 1.0; do
    echo "--- tune lam_concept=$LAMC lr=$LR  $(date '+%T')" >> "$LOG"
    T2_LAM_CONCEPT=$LAMC T3_LR=$LR T3_SEEDS=0 T3_TAG="K32_lc${LAMC}_lr${LR}_tune" \
      nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
    RC=$?
    echo "    tune lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
    [ "$RC" -ne 0 ] && fail "tuning failed for lam_concept=$LAMC lr=$LR (rc=$RC)"
  done
done

# guard: the picker must actually have files to read
ls runs/t3/models_K32_lc*_tune.json >/dev/null 2>&1 \
  || fail "no tuning files produced — refusing to continue"

# ---- apply the frozen joint selection rule ----
echo "--- picking lr per lam_concept  $(date '+%T')" >> "$LOG"
"$PY" scripts/t3_lamc_pick.py >> "$LOG" 2>&1
echo "    pick rc=$?" >> "$LOG"

# ---- phase 2: 3 seeds at each lam_concept's selected lr ----
while read -r LAMC LR; do
  [ -z "$LAMC" ] && continue
  echo "--- confirm lam_concept=$LAMC lr=$LR (3 seeds)  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT=$LAMC T3_LR=$LR T3_SEEDS=0,1,2 T3_TAG="K32_lc${LAMC}_lr${LR}_sel" \
    nice -n 19 ionice -c3 "$PY" scripts/t3_models.py >> "$LOG" 2>&1
  RC=$?
  echo "    confirm lc=$LAMC lr=$LR rc=$RC" >> "$LOG"
  [ "$RC" -ne 0 ] && fail "confirmation failed for lam_concept=$LAMC lr=$LR (rc=$RC)"
done < <("$PY" -c "
import json
s=json.load(open('runs/t3/lamc_selection.json'))
for k,v in s.items(): print(v['lam_concept'], v['lr'])
")

# ---- phase 3: audit with the supervised/unsupervised split ----
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
  [ "$RC" -ne 0 ] && fail "audit failed for lam_concept=$LAMC lr=$LR (rc=$RC)"
done < <("$PY" -c "
import json
s=json.load(open('runs/t3/lamc_selection.json'))
for k,v in s.items(): print(v['lam_concept'], v['lr'])
")

echo "=== step 5 done  $(date '+%F %T') ===" >> "$LOG"