#!/bin/bash
# IS THE GENERALISATION FAILURE DATA-STARVATION OR CAPACITY? ONE CHANGE (2026-11-20 21:10)
#
# MEASURED (3 seeds, 6000 steps, 600 rule pairs): held-out rule accuracy 35.00 / 34.33 / 40.83 (mean 36.72), train 52.50 / 53.44 /
# 55.94 (mean 53.96). A 17.2-point gap, and four times the training from 1500 steps bought only +11.72 points. That pattern fits
# 600 pairs being memorised, not a rule being learned.
#
# THE ONE CHANGE THIS RUN MAKES: n_rule_train 600 -> 5000, out of 10,000 possible ordered pairs. Steps, model shape, optimiser,
# lr, fact count, seeds and scoring are IDENTICAL. Nothing else moves.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 the model CAN generalise when given data: held-out rule accuracy above 60% on all three seeds (vs mean 36.72%).
#   P2 the train/held-out gap shrinks below 10 points (from 17.24), which is what learning-rather-than-memorising looks like.
#   P3 the switch keeps working: facts 100% with the store, collisions <= 1%, rule accuracy changed by <= 1.0 point.
# FALSIFIER: held-out rule accuracy stays below 45% even with 5000 pairs -> data is NOT the binding constraint. Generalisation at this
#   scale then needs an architectural change (a scratchpad / loop over output digits), which gets its own pre-registration. It does NOT
#   get rescued by widening this model inside this run.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/MEMG-DATA-${RUN_TAG}.txt
: >> "$OUT"
echo "=== 5000 rule pairs, 6000 steps, 3 seeds, $(date '+%F %T') ===" | tee -a "$OUT"
echo "reference (600 pairs, same steps): held-out mean 36.72%, train mean 53.96%, gap 17.24" | tee -a "$OUT"
for S in 0 1 2; do
  LOG="lens/runs/chain_memgdata_s${S}_${RUN_TAG}.out"
  echo "--- seed $S ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/memg.py --steps 6000 --batch 32 --d 64 --layers 2 --heads 4 --seed "$S" \
    --n-rule-train 5000 > "$LOG" 2>&1
  RC=$?
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then echo "  seed $S CRASHED rc=$RC" | tee -a "$OUT"; tail -4 "$LOG" | tee -a "$OUT"; exit 1; fi
  grep -E '^data:|^rule accuracy|^fact accuracy|switch hits|step 6000/6000|OUTCOME' "$LOG" | tee -a "$OUT"
done
echo "MEMG-DATA COMPLETE" | tee -a "$OUT"
