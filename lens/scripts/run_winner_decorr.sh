#!/bin/bash
# THE ONE ARM THE CLOSURE LEAVES OPEN: does the most accurate configuration improve with decorrelation? (2026-11-20 13:13)
#
# MEASURED BASIS: 36 concepts + direct path at lam_decorr 0 is the most accurate configuration (acc 12.18%, ppl 712.92, audit 25/36,
# supply 8.476x). The decorrelation weights 0.2/0.5/1.0 were only ever measured on the BOTTLENECKED shape (no direct path), where they
# cut cancellation 236x -> 38x and saturated at 0.5. Whether the winner benefits has never been tested.
#
# THE ONE CHANGE against that 12.18% arm: --lam-decorr 0.5. Everything else identical (36 concepts via TC_WIDE default, direct ON,
# 100k windows, 1 epoch, seed 0, same method).
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 accuracy >= 12.68% (12.18 + 0.5), OR within 0.5 points of 12.18 while positive supply falls below 8.476x. Either is a
#      reason to keep decorrelation in the best configuration.
# FALSIFIER: accuracy more than 1.0 point below 12.18% -> decorrelation HURTS the direct-path configuration, the best-known setting
#   stays lam_decorr 0, and NO further concept arm of any kind is run. The concept line is then closed in full.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
MAXW=${MAXW:-100000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/WINNER-DECORR.txt}
: > "$OUT"
echo "=== winner + decorrelation 0.5, $(date '+%F %T') MAXW=$MAXW ===" | tee -a "$OUT"
echo "reference: 36 concepts + direct, decorr 0 = acc 12.18%, ppl 712.92, audit 25/36, supply 8.476x" | tee -a "$OUT"
env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 --lam-decorr 0.5 \
  --concept-target window --method ours --seed 0 --max-windows "$MAXW" --direct --out "windec_m${MAXW}w" \
  >> lens/runs/chain_windec.out 2>&1
RC=$?
echo "  rc=$RC" | tee -a "$OUT"
[ "$RC" -ne 0 ] && { echo "  ARM FAILED rc=$RC" | tee -a "$OUT"; exit 1; }
grep -E 'final VAL' lens/runs/chain_windec.out | tail -1 | tee -a "$OUT"
env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 tools/concept_rank.py --ckpt "lens/runs/chat/windec_m${MAXW}w_ep${EPOCHS}.npz" \
  --windows 100 --fixed-runner --out "lens/runs/WINNER-DECORR.json" 2>&1 \
  | grep -E 'positive supply|COUNT-INVARIANT|entropy|max \|sum' | tee -a "$OUT"
echo "WINNER-DECORR COMPLETE" | tee -a "$OUT"
