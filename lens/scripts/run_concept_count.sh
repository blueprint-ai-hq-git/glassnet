#!/bin/bash
# IS THE RESIDUAL CANCELLATION A FUNCTION OF CONCEPT COUNT? (2026-11-20 13:08)
#
# The decorrelation sweep closed by pre-registration: supply 236.039x -> 62.600x -> 38.051x -> 38.356x, so decorrelation saturates at
# 0.5 and the remaining 38x is STRUCTURAL. The pre-registered remedy is "fewer, wider concepts", and it turns out no code change is
# needed to test the count half of that: train_chat.py line 939 selects the catalogue by env (`TC_WIDE=1` -> chat_concepts_wide, the
# 48-concept set that survives support filtering as the 36 we have been measuring; `TC_WIDE=0` -> chat_concepts, the original 9).
#
# THE ONE CHANGE against the d05 arm: the concept catalogue (36 concepts -> 9). lam_decorr stays 0.5, data, seed, budget, width and
# method are identical, so the comparison is paired against d05's published numbers (acc 7.19%, supply 38.051x).
# CONFOUND, STATED UP FRONT: the 9-concept catalogue also has BROADER definitions, so count and breadth move together. That is what
# "fewer, wider" means, so this is the remedy as specified - but it is not a pure count intervention, and no claim here will be
# phrased as one.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (measured by concept_rank.py --fixed-runner):
#   P1 with 9 concepts the POSITIVE SUPPLY falls to at least 20% below d05's 38.051x (i.e. <= 30.44x). That is what "the count is
#      the cause" looks like.
#   P2 accuracy stays within 2.0 points of d05's 7.19%.
# FALSIFIER: if supply stays within 20% of 38.051x, then cancellation is NOT a function of concept count. The structural remedy
#   fails, the concept-bottleneck line CLOSES for good, and no further concept-count or strength arm is run.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
MAXW=${MAXW:-100000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/CONCEPT-COUNT.txt}
: > "$OUT"
echo "=== is cancellation a function of concept count? $(date '+%F %T') MAXW=$MAXW ===" | tee -a "$OUT"
echo "reference (d05, 36 concepts, lam_decorr 0.5): acc 7.19%, supply 38.051x" | tee -a "$OUT"

env -u PYTHONPATH -u PYTHONHOME TC_WIDE=0 nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 --lam-decorr 0.5 \
  --concept-target window --method ours --seed 0 --max-windows "$MAXW" --out "narrow_m${MAXW}w" \
  >> lens/runs/chain_narrow.out 2>&1
RC=$?
echo "  rc=$RC" | tee -a "$OUT"
[ "$RC" -ne 0 ] && { echo "  ARM FAILED rc=$RC" | tee -a "$OUT"; exit 1; }
grep -E 'final VAL' lens/runs/chain_narrow.out | tail -1 | tee -a "$OUT"
echo "  concept count actually used:" | tee -a "$OUT"
grep -oE 'concepts [0-9]+/[0-9]+' lens/runs/chain_narrow.out | tail -1 | tee -a "$OUT"

env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
  .venv-gpu/bin/python3 tools/concept_rank.py --ckpt "lens/runs/chat/narrow_m${MAXW}w_ep${EPOCHS}.npz" \
  --windows 100 --fixed-runner --out "lens/runs/CONCEPT-COUNT.json" 2>&1 \
  | grep -E 'positive supply|CONCENTRATION|max \|sum|windows:' | tee -a "$OUT"
echo "CONCEPT-COUNT COMPLETE" | tee -a "$OUT"
