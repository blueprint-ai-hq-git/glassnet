#!/bin/bash
# NARROW CONCEPTS + THE DIRECT PATH: the first configuration that might be both honest and accurate (2026-11-20 13:10)
#
# MEASURED BASIS: the 36-concept arm WITH the direct path reached acc 12.18% (ppl 712.92, concepts 25/36) while every bottlenecked
# arm sits near 7.4%. Separately, the 9-concept bottlenecked arm reached 7.37% with EVERY concept passing audit (9/9) and
# cancellation nearly halved (38.051x -> 19.839x). So the two good properties have never been combined: honesty came from fewer
# concepts, accuracy came from the direct path.
#
# THE ONE CHANGE against the 36-concept direct-path arm: the catalogue (TC_WIDE=0 -> 9 concepts). Direct path ON, lam_decorr 0.5,
# data, seed, budget, width and method identical.
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 accuracy beats the 36-concept direct-path arm's 12.18% by at least +1.0 point, OR matches it within 1.0 point while the audit
#      stays at 8/9 concepts passing. Either is a win: better, or equally accurate but fully honest.
#   P2 positive supply stays below the 36-concept direct-path arm's 8.476x (i.e. the cancellation improvement survives the direct path).
# FALSIFIER: if accuracy falls more than 2.0 points below 12.18% AND the audit drops below 6 of 9, then narrowing does not combine
#   with the direct path and the honest+accurate configuration does not exist at this scale - the concept line closes for good.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
MAXW=${MAXW:-100000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/NARROW-DIRECT.txt}
: > "$OUT"
echo "=== narrow concepts + direct path, $(date '+%F %T') MAXW=$MAXW ===" | tee -a "$OUT"
echo "reference: 36 concepts + direct = acc 12.18%, ppl 712.92, audit 25/36, supply 8.476x" | tee -a "$OUT"

env -u PYTHONPATH -u PYTHONHOME TC_WIDE=0 nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 --lam-decorr 0.5 \
  --concept-target window --method ours --seed 0 --max-windows "$MAXW" --direct --out "narrowdir_m${MAXW}w" \
  >> lens/runs/chain_narrowdir.out 2>&1
RC=$?
echo "  rc=$RC" | tee -a "$OUT"
[ "$RC" -ne 0 ] && { echo "  ARM FAILED rc=$RC" | tee -a "$OUT"; exit 1; }
grep -E 'final VAL' lens/runs/chain_narrowdir.out | tail -1 | tee -a "$OUT"
grep -oE 'concepts [0-9]+/[0-9]+' lens/runs/chain_narrowdir.out | tail -1 | tee -a "$OUT"
env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 tools/concept_rank.py --ckpt "lens/runs/chat/narrowdir_m${MAXW}w_ep${EPOCHS}.npz" \
  --windows 100 --fixed-runner --out "lens/runs/NARROW-DIRECT.json" 2>&1 \
  | grep -E 'positive supply|max \|sum|windows:' | tee -a "$OUT"
echo "NARROW-DIRECT COMPLETE" | tee -a "$OUT"
