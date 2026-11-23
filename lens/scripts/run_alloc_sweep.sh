#!/bin/bash
# CAPACITY ALLOCATION: LET THE MEASUREMENT DECIDE WHERE THE PARAMETER BUDGET GOES (2026-11-20 18:50)
#
# THE ASK: "automatically parameters allocation". Today the split between the token table, the blocks and the depth is chosen by hand.
# This run makes the choice from evidence instead: at a roughly constant parameter budget, vary where the capacity sits, and let the
# held-out number pick. This is automatic allocation as MEASUREMENT-DRIVEN SELECTION - not yet growth or pruning during training, and
# it is labelled that way deliberately so it cannot be mistaken for the harder thing.
#
# THE SHIPPING OPTIMISER IS USED THROUGHOUT (momentum, lr 0.05) because step-size automation was abandoned by its own pre-registered
# falsifier (gain rule: 8.90 / 5.89 against a 12.51% reference). One thing changes at a time.
#
# CONFIGS (all seed 0, 300 steps, 20k windows - 300 not 600 so four arms fit one wake):
#   A  rank 32  d 64  layers 2   <- the hand-chosen default, 454,336 params
#   B  rank  8  d 96  layers 3   <- more depth, far smaller table
#   C  rank 16  d 96  layers 2   <- middle table, wider blocks
#   D  rank  8  d 128 layers 2   <- smallest table, widest blocks
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 the best config beats the hand-chosen default A by at least +0.5 held-out points.
#   P2 at least one config with FEWER total parameters than A also beats A -> the default is wasting capacity.
# FALSIFIER: no config beats A -> the hand-chosen split is already at or near the best available at this budget, allocation has nothing
#   to win here, and this sweep stops (no widening of the grid, no second seed hunting for a winner).
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/ALLOC-SWEEP-${RUN_TAG}.txt
: >> "$OUT"
echo "=== capacity allocation sweep, $(date '+%F %T') ===" | tee -a "$OUT"
run_cfg() {   # label rank d layers
  local LOG="lens/runs/chain_alloc_${1}_${RUN_TAG}.out"
  echo "--- $1 (rank=$2 d=$3 layers=$4) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    ${PYTHON:-python3} -u lens/glass_lm.py \
    --steps 300 --batch 16 --T 32 --d "$3" --layers "$4" --heads 4 --d-mlp "$((2*$3))" --windows 20000 \
    --emb-rank "$2" --seed 0 --opt momentum --lr 0.05 --log-every 300 > "$LOG" 2>&1
  local RC=$?
  local OUTCOME
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then
    echo "  $1 CRASHED (rc=$RC) - stopping" | tee -a "$OUT"; tail -4 "$LOG" | tee -a "$OUT"; return 1
  fi
  echo "  outcome=$OUTCOME rc=$RC" | tee -a "$OUT"
  grep -E 'model:|OUTPUT-TABLE SHARE|HELD-OUT' "$LOG" | tail -3 | tee -a "$OUT"
  return 0
}
run_cfg A_default   32 64  2 || exit 1
run_cfg B_deep_small 8 96  3 || exit 1
run_cfg C_mid_wide  16 96  2 || exit 1
run_cfg D_wide_small 8 128 2 || exit 1
echo "--- P1/P2: compare the four HELD-OUT lines and the four parameter totals ---" | tee -a "$OUT"
echo "ALLOC-SWEEP COMPLETE" | tee -a "$OUT"
