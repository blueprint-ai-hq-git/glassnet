#!/bin/bash
# e_run.sh — TARGET E: K=9, ALL NINE taught (zero unnamed), extended concept set.
# One change from Target D: 6 concepts -> 9. Floor unchanged, bars unchanged.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=9
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T2_CONCEPT_SET=meaningful_ext
export T2_N_TAUGHT=9
export T3_CKPT_TAG=e
PY=".venv-gpu/bin/python3"
LOG=runs/t3/e_run.log
fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }

[ -f runs/t3/meaningful_targets_ext.json ] || fail "no extended targets"
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing"

echo "=== TARGET E (K=9, all taught) — TUNING  $(date '+%F %T') ===" | tee -a "$LOG"
for lc in 0.1 0.5 1.0; do
  for lr in 0.1 1.0; do
    TAG="e_scr_lc${lc}_lr${lr}"
    J="runs/t3/models_${TAG}.json"
    N=$("$PY" -c "
import json,os
f='$J'
print(0 if not os.path.exists(f) else len([x for x in json.load(open(f)) if x['model']=='readable']))
" 2>/dev/null || echo 0)
    if [ "$N" = "1" ]; then echo "--- skip $TAG" >> "$LOG"; continue; fi
    echo "--- screen lc=$lc lr=$lr seed0  $(date '+%T')" | tee -a "$LOG"
    T2_LAM_CONCEPT="$lc" T3_LR="$lr" T3_SEEDS=0 T3_MODELS=readable T3_TAG="$TAG" \
      nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
    [ $? -ne 0 ] && echo "!!! $TAG failed" | tee -a "$LOG"
  done
done

echo "--- TOP 2 from the screen (frozen rule)" | tee -a "$LOG"
"$PY" scripts/t3_e_top2.py 2>&1 | tee -a "$LOG"
[ $? -ne 0 ] && fail "top2 failed"

for combo in $("$PY" -c "
import json; d=json.load(open('runs/t3/e_screen.json'))
print(' '.join(f\"{t['lc']}:{t['lr']}\" for t in d['top2']))"); do
  LC="${combo%%:*}"; LR="${combo##*:}"
  echo "--- confirm lc=$LC lr=$LR seeds 1,2  $(date '+%T')" | tee -a "$LOG"
  T2_LAM_CONCEPT="$LC" T3_LR="$LR" T3_SEEDS=1,2 T3_MODELS=readable \
    T3_TAG="e_cfm_lc${LC}_lr${LR}" \
    nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
  [ $? -ne 0 ] && fail "confirm lc=$LC lr=$LR failed"
done

echo "--- winner on the 3-seed mean" | tee -a "$LOG"
"$PY" scripts/t3_e_pick.py 2>&1 | tee -a "$LOG"
[ $? -ne 0 ] && fail "selection failed"

LR=$("$PY" -c "import json;print(json.load(open('runs/t3/e_selection.json'))['lr'])")
LC=$("$PY" -c "import json;print(json.load(open('runs/t3/e_selection.json'))['lc'])")
echo "--- winner lr=$LR lc=$LC" | tee -a "$LOG"

echo "--- matched normal: SAME K=9, SAME params, lam_concept=0, 3 seeds" | tee -a "$LOG"
T2_LAM_CONCEPT=0.0 T3_LR="$LR" T3_SEEDS=0,1,2 T3_MODELS=normal T3_TAG="e_normal" \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "normal training failed"

echo "=== TARGET E TRAINING DONE  $(date '+%F %T') ===" | tee -a "$LOG"