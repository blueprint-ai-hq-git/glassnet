#!/bin/bash
# b_grid.sh — TARGET B: the FULL grid, no screening. 12 readable + 4 normal combos,
# every one on all 5 seeds = 80 runs. Resumable: a combo whose JSON already holds 5
# readable/normal records is skipped, so an interruption costs one run, not the grid.
#
# T3_CKPT_TAG=b keeps B's checkpoints distinct from A3's at the same lr/lc/K. A3's audit
# reads the untagged files, so without the tag B would silently break A3's reproducibility.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=4          # machine stays responsive; owner may be home
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=16
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T2_CONCEPT_SET=clusters16
export T2_N_TAUGHT=10
export T3_CKPT_TAG=b
export T3_SEEDS=0,1,2,3,4
PY=".venv-gpu/bin/python3"
LOG=runs/t3/b_run.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing"
[ -f runs/t3/concept_clusters16.json ] || fail "no k=16 targets"

# safety: keep a copy of the pre-B checkpoints (87 MB) so nothing is lost either way
if [ ! -d runs/t3/ckpt_preB ]; then
  cp -a runs/t3/ckpt runs/t3/ckpt_preB && echo "backed up ckpt -> ckpt_preB" | tee -a "$LOG"
fi

echo "=== TARGET B GRID  $(date '+%F %T')  (12 readable + 4 normal combos x 5 seeds = 80 runs)" \
  | tee -a "$LOG"

for lc in 0.1 0.5 1.0; do
  for lr in 0.1 0.5 1.0 2.0; do
    TAG="b_read_lc${lc}_lr${lr}"
    J="runs/t3/models_${TAG}.json"
    N=$("$PY" -c "
import json,os
f='$J'
print(0 if not os.path.exists(f) else len([x for x in json.load(open(f)) if x['model']=='readable']))
" 2>/dev/null || echo 0)
    if [ "$N" = "5" ]; then echo "--- skip $TAG (already 5 seeds)" >> "$LOG"; continue; fi
    echo "--- readable lc=$lc lr=$lr  $(date '+%T')" | tee -a "$LOG"
    T2_LAM_CONCEPT="$lc" T3_LR="$lr" T3_MODELS=readable T3_TAG="$TAG" \
      nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
    [ $? -ne 0 ] && echo "!!! $TAG failed, continuing with the rest" | tee -a "$LOG"
  done
done

for lr in 0.1 0.5 1.0 2.0; do
  TAG="b_norm_lr${lr}"
  J="runs/t3/models_${TAG}.json"
  N=$("$PY" -c "
import json,os
f='$J'
print(0 if not os.path.exists(f) else len([x for x in json.load(open(f)) if x['model']=='normal']))
" 2>/dev/null || echo 0)
  if [ "$N" = "5" ]; then echo "--- skip $TAG (already 5 seeds)" >> "$LOG"; continue; fi
  echo "--- normal lr=$lr  $(date '+%T')" | tee -a "$LOG"
  T3_LR="$lr" T3_MODELS=normal T3_TAG="$TAG" \
    nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
  [ $? -ne 0 ] && echo "!!! $TAG failed, continuing with the rest" | tee -a "$LOG"
done

echo "=== TARGET B GRID DONE  $(date '+%F %T') ===" | tee -a "$LOG"