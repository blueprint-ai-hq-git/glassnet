#!/bin/bash
# CAN THE MODEL ACTUALLY GENERALISE THE RULE? (2026-11-20 20:35)
#
# WHAT THE LAST RUN ESTABLISHED: the memorize/generalise SWITCH works on all 3 seeds (facts 8-13.5% -> 100%, zero collisions, rule
# accuracy +0.00). WHAT IT DID NOT ESTABLISH: that the model can apply a rule at all - held-out addition accuracy was only 23-26%
# while train accuracy was 49.7-52.5%, i.e. the rule half was never learned at 1500 steps.
#
# So this run tests the missing half with the single change that the diagnosis calls for: MORE TRAINING on the same trap dataset,
# same model shape, same protocol. 6000 steps instead of 1500, 3 seeds.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 the model CAN generalise: held-out rule accuracy rises above 60% on all three seeds (vs 23-26% at 1500 steps).
#   P2 the switch still behaves: fact accuracy 100% with the switch, collisions <= 1%, rule accuracy changed by <= 1.0 point.
# FALSIFIER: held-out rule accuracy stays below 40% even at 6000 steps -> this architecture at this width cannot learn two-digit
#   addition within this budget. The generalisation half of the ask is then NOT demonstrated, and it gets reported as a failure of the
#   generalisation side rather than buried under the switch's success. No rescue by widening the model without a new pre-registration.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/MEMG-LONG-${RUN_TAG}.txt
: >> "$OUT"
echo "=== can the model learn the rule? 6000 steps, 3 seeds, $(date '+%F %T') ===" | tee -a "$OUT"
for S in 0 1 2; do
  LOG="lens/runs/chain_memglong_s${S}_${RUN_TAG}.out"
  echo "--- seed $S ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/memg.py --steps 6000 --batch 32 --d 64 --layers 2 --heads 4 --seed "$S" > "$LOG" 2>&1
  RC=$?
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then echo "  seed $S CRASHED rc=$RC" | tee -a "$OUT"; tail -4 "$LOG" | tee -a "$OUT"; exit 1; fi
  grep -E '^rule accuracy|^fact accuracy|switch hits|^P1 |^P2 |^P3 |OUTCOME' "$LOG" | tee -a "$OUT"
  grep -E 'step 6000/6000' "$LOG" | tee -a "$OUT"
done
echo "MEMG-LONG COMPLETE" | tee -a "$OUT"
