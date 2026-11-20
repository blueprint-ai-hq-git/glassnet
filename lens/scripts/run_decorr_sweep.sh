#!/bin/bash
# DOES CANCELLATION KEEP FALLING WITH STRONGER DECORRELATION? (2026-11-20 13:02)
#
# MEASURED BASIS: at lam_decorr 0.2 positive supply fell 236.039x -> 62.600x the net margin (-73.5%) for -0.06 accuracy points.
# The control reproduces the independent 100k arm to three decimals (953.87 ppl / 7.31% / 236.03x), so it is DETERMINISTIC and a
# fresh control in this run is not required - it is reused from lens/runs/chat/dec_ctrl_m100000w_ep1.npz and that reuse is stated.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (all measured by concept_rank.py --fixed-runner):
#   P1 cancellation keeps falling: supply(1.0) < supply(0.5) < control 236.039x, and supply(1.0) is at least another 20% below
#      supply(0.5). That is what "decorrelation is a lever with headroom" means.
#   P2 accuracy stays within 2.0 points of the 7.31% control at both strengths.
# FALSIFIER: if supply(1.0) is NOT at least 20% below supply(0.5), decorrelation SATURATES at 0.2 and the residual cancellation is
#   structural - the remedy then is FEWER, WIDER concepts (a structural change), and this line closes rather than sweeping further.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
MAXW=${MAXW:-100000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/DECORR-SWEEP.txt}
: > "$OUT"
echo "=== decorrelation sweep 0.5 / 1.0 vs the deterministic control, $(date '+%F %T') MAXW=$MAXW ===" | tee -a "$OUT"
echo "control (reused, deterministic): dec_ctrl_m${MAXW}w_ep${EPOCHS}, acc 7.31%, supply 236.039x" | tee -a "$OUT"

for LAM in 0.5 1.0; do
  TAG="d$(echo $LAM | tr -d '.')"
  echo "--- $TAG (lam_decorr=$LAM) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 --lam-decorr "$LAM" \
    --concept-target window --method ours --seed 0 --max-windows "$MAXW" --out "dec_${TAG}_m${MAXW}w" \
    >> "lens/runs/chain_dec_$TAG.out" 2>&1
  RC=$?
  echo "  rc=$RC" | tee -a "$OUT"
  [ "$RC" -ne 0 ] && { echo "  ARM $TAG FAILED" | tee -a "$OUT"; exit 1; }
  grep -E 'final VAL' "lens/runs/chain_dec_$TAG.out" | tail -1 | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
    .venv-gpu/bin/python3 tools/concept_rank.py --ckpt "lens/runs/chat/dec_${TAG}_m${MAXW}w_ep${EPOCHS}.npz" \
    --windows 100 --fixed-runner --out "lens/runs/DECORR-SWEEP-$TAG.json" 2>&1 \
    | grep -E 'positive supply|CONCENTRATION|max \|sum' | tee -a "$OUT"
done
echo "DECORR-SWEEP COMPLETE" | tee -a "$OUT"
