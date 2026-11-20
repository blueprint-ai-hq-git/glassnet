#!/bin/bash
# THE OWED PER-ELEMENT RETRY, AND A RUNNER THAT CANNOT ABORT ON A RESULT (2026-11-20 17:58)
#
# WHAT HAPPENED IN THE LAST TWO RUNS, both runner defects of mine:
#   run 1: a diverged arm (rc 1 at the time) aborted the two adaptive arms - the actual test.
#   run 2: rc 3 was handled, but an arm that COMPLETED with a bad result (rc 1, "below constant") still aborted the run at arm 3.
# The root cause is the same: the trainer used a nonzero exit code for both "bad result" and "crash", so the runner could not tell
# them apart. Now the trainer prints an explicit OUTCOME line (BEATS_CONSTANT / BELOW_CONSTANT / DIVERGED) and this runner aborts
# ONLY when that line is missing, i.e. when the arm really did crash.
#
# MEASURED SO FAR (all at emb_rank 32, seed 0, 600 steps):
#   momentum  lr 0.05  -> 12.51%   (reference)
#   momentum  lr 0.50  -> DIVERGED (loss nan by step 300)
#   adaptive  lr 0.50  -> 0.03%    (loss climbed 9.41 -> 27.57; per-tensor scale did NOT rescue the high lr)
# So P2 FAILED and the pre-registered falsifier owes exactly ONE variant: per-ELEMENT RMS. That is what this run tests.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 (the owed retry) per-element adaptive at lr 0.50 lands within 1.0 point of 12.51%.
#   P2 per-element at lr 0.05 is itself near the reference (>= 11.5%), so the variant is not simply broken everywhere.
# FALSIFIER: per-element at lr 0.50 is also at least 2.0 points below 12.51% -> gradient-magnitude normalisation of ANY granularity
#   does not remove lr sensitivity. The direction CLOSES, and the next mechanism is loss-RESPONSE step control (a gain rule that
#   grows or shrinks the step from the measured change in loss), pre-registered separately. No further normalisation variants.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/GLASSLM-OPT-${RUN_TAG}.txt
: >> "$OUT"
echo "=== per-element adaptive retry, $(date '+%F %T') ===" | tee -a "$OUT"
run_arm() {   # label opt lr
  local LOG="lens/runs/chain_glopt2_${1}_${RUN_TAG}.out"
  echo "--- $1 (opt=$2 lr=$3) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/glass_lm.py \
    --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 \
    --emb-rank 32 --seed 0 --opt "$2" --lr "$3" --log-every 300 > "$LOG" 2>&1
  local RC=$?
  local OUTCOME
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then
    echo "  $1 CRASHED with no outcome line (rc=$RC) - this IS a failure, stopping" | tee -a "$OUT"
    tail -5 "$LOG" | tee -a "$OUT"
    return 1
  fi
  echo "  outcome=$OUTCOME  rc=$RC" | tee -a "$OUT"
  grep -E 'optimiser:|HELD-OUT' "$LOG" | tail -2 | tee -a "$OUT"
  return 0
}
run_arm adaElem_lr050 adaptive_elem 0.50 || exit 1
run_arm adaElem_lr005 adaptive_elem 0.05 || exit 1
run_arm adaElem_lr500 adaptive_elem 5.00 || exit 1
echo "--- P1/P2: compare the three HELD-OUT lines above (reference 12.51%) ---" | tee -a "$OUT"
echo "GLASSLM-OPT2 COMPLETE" | tee -a "$OUT"
