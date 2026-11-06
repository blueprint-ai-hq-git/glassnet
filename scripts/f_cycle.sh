#!/bin/bash
# f_cycle.sh — TARGET F step 2: train every student on generated data only, evaluate on REAL val.
# Arms: rule {all, conf50} x kind {normal, readable} x 3 seeds. Same budget across arms.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export F_EPOCHS=40
export F_PATIENCE=10
export F_TAG=f
PY=".venv-gpu/bin/python3"
LOG=runs/t3/f_cycle.log
fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }

for rule in all conf50; do
  [ -f "runs/t3/f_corpus_${rule}.npz" ] || fail "missing corpus for rule $rule"
done
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing"

echo "=== TARGET F — STUDENTS  $(date '+%F %T') ===" | tee -a "$LOG"
for rule in all conf50; do
  for kind in normal readable; do
    for seed in 0 1 2; do
      J="runs/t3/models_f_f_${kind}_${rule}.json"
      N=$("$PY" -c "
import json,os
f='$J'
print(0 if not os.path.exists(f) else len([x for x in json.load(open(f)) if x['seed']==$seed]))
" 2>/dev/null || echo 0)
      if [ "$N" = "1" ]; then echo "--- skip $kind/$rule/seed$seed" >> "$LOG"; continue; fi
      echo "--- $kind rule=$rule seed=$seed  $(date '+%T')" | tee -a "$LOG"
      if [ "$kind" = "readable" ]; then LC=0.1; else LC=0.0; fi
      F_KIND="$kind" F_RULE="$rule" F_SEED="$seed" F_LR=1.0 F_LC="$LC" \
        nice -n 19 ionice -c3 "$PY" -u scripts/f_train.py >> "$LOG" 2>&1
      [ $? -ne 0 ] && echo "!!! $kind/$rule/seed$seed failed" | tee -a "$LOG"
    done
  done
done
echo "=== TARGET F STUDENTS DONE  $(date '+%F %T') ===" | tee -a "$LOG"