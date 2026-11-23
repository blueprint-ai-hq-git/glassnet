#!/bin/bash
# THE MECHANISM THE DIAGNOSIS POINTS AT: step size chosen by MEASURED loss response (2026-11-20 18:05)
#
# WHY BOTH EARLIER ATTEMPTS FAILED, per the measurement: normalising a gradient by its own magnitude throws away how much the loss
# cares about each coordinate. Per-tensor: 0.03% at lr 0.50. Per-element: 0.26%, 0.00%, 0.01% at lr 0.50 / 0.05 / 5.00 - broken at
# EVERY scale, because every coordinate gets an equal-size step. Both variants are the wrong family.
#
# THE GAIN RULE: keep ONE global direction (gradient normalised once), then choose the scalar step by trial - take the step, measure
# the loss, keep it if the loss fell, otherwise halve and retry; grow 1.3x after a first-try accept. The initial value is only a guess.
# Smoke-tested before this launch: 20 steps, loss 9.38 -> 8.10, step size self-adjusted 0.5 -> 2.0 -> 1.25, and it beat its constant.
#
# THE TEST: the SAME two initial values that defeated the hand-set and normalisation optimisers.
#   1 gain starting at 0.50  (fixed lr 0.50 diverged; per-tensor 0.50 sank to 0.03%)
#   2 gain starting at 0.005 (a step 100x smaller than the working constant - the opposite direction)
# Reference to beat: momentum at lr 0.05 = 12.51% (emb_rank 32, seed 0, 600 steps).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 gain from 0.50 lands within 1.0 point of 12.51%.
#   P2 gain from 0.005 ALSO lands within 1.0 point of 12.51% -> the starting value stops mattering, which is the property being sought.
# FALSIFIER: either arm at least 2.0 points below 12.51% -> loss-response step control also fails at this scale. Automation of the
#   step size is then ABANDONED for this model (hand-set lr documented as the shipping choice), and the effort moves to capacity
#   allocation, which was the next ask anyway. No third optimiser variant.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/GLASSLM-GAIN-${RUN_TAG}.txt
: >> "$OUT"
echo "=== gain rule: step size from measured loss response, $(date '+%F %T') ===" | tee -a "$OUT"
echo "reference: momentum lr 0.05 = 12.51% held-out" | tee -a "$OUT"
run_arm() {   # label initial_step
  local LOG="lens/runs/chain_gain_${1}_${RUN_TAG}.out"
  echo "--- $1 (opt=gain start=$2) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    ${PYTHON:-python3} -u lens/glass_lm.py \
    --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 \
    --emb-rank 32 --seed 0 --opt gain --lr "$2" --log-every 150 > "$LOG" 2>&1
  local RC=$?
  local OUTCOME
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then
    echo "  $1 CRASHED with no outcome line (rc=$RC) - stopping" | tee -a "$OUT"; tail -5 "$LOG" | tee -a "$OUT"; return 1
  fi
  echo "  outcome=$OUTCOME rc=$RC" | tee -a "$OUT"
  grep -E 'initial step|HELD-OUT' "$LOG" | tail -2 | tee -a "$OUT"
  grep -E 'step  (600|150|300|450)/600' "$LOG" | tail -2 | tee -a "$OUT"
  return 0
}
run_arm gain_from050 0.50 || exit 1
run_arm gain_from0005 0.005 || exit 1
echo "--- P1/P2: compare both HELD-OUT lines against the 12.51% reference ---" | tee -a "$OUT"
echo "GLASSLM-GAIN COMPLETE" | tee -a "$OUT"
