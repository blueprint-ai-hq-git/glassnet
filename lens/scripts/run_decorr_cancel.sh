#!/bin/bash
# DOES DECORRELATION REMOVE THE CANCELLATION? (2026-11-20 13:40)
#
# THE MEASURED FACT THIS ATTACKS. With the runner-up held fixed, the concept layer's exact attribution shows 17 of 36 concepts
# supplying positive margin and 19 OPPOSING, with the positive supply running 16.43x the net margin on the --direct arm and 9.45x
# on the bottlenecked arm. The answer is the small residue of large opposing pushes, which is why explain() was exact on 0 of 200
# windows and why the old share metric could be made to say 46% when the net flow was 1.3%.
#
# THE ONE CHANGE: `--lam-decorr 0.2` against a paired control at 0. It is already in the trainer, and the image model measured a
# -0.307 decorrelation at this value. Nothing else differs: same data, seed, budget, width, window, method.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (both read by tools/concept_rank.py --fixed-runner, the exact instrument):
#   P1 the decorrelated arm's POSITIVE SUPPLY (as a multiple of the net margin) falls by at least 20% relative to its control.
#      This is the number that decides whether the layer can ever explain itself.
#   P2 the decorrelated arm stays within 2.0 accuracy points of its control. Structure bought by more than that is a bad trade.
# FALSIFIER: if positive supply does not fall by 20% relative, the cancellation is INTRINSIC to a 36-concept layer trained by
#   cross-entropy. The remedy then is FEWER, WIDER concepts - a structural change - and the decorrelation line closes rather than
#   being refined at other weights (the strength family is already measured four ways elsewhere in this project).
#
# DO NOT LAUNCH WHILE BOTH CORE PAIRS ARE BUSY: off2pair owns 6,7 and customarch3 owns 8,9. A third trainer would slow both and
# make their paired comparisons unpinnable. Launched by the next wake that finds a pair free.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
MAXW=${MAXW:-100000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/DECORR-CANCEL.txt}
: > "$OUT"
echo "=== does decorrelation remove the cancellation? $(date '+%F %T') MAXW=$MAXW ===" | tee -a "$OUT"

run_arm() {   # $1 label, $2 lam_decorr
  echo "--- $1 (lam_decorr=$2) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 --lam-decorr "$2" \
    --concept-target window --method ours --seed 0 --max-windows "$MAXW" --out "dec_$1_m${MAXW}w" \
    >> "lens/runs/chain_dec_$1.out" 2>&1
  local rc=$?
  echo "  rc=$rc" | tee -a "$OUT"
  if [ "$rc" -ne 0 ]; then echo "  ARM $1 FAILED rc=$rc" | tee -a "$OUT"; return 1; fi
  grep -E 'final VAL' "lens/runs/chain_dec_$1.out" | tail -1 | tee -a "$OUT"
  # the exact instrument that decides P1 - run on this arm's own checkpoint
  env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 8,9 \
    .venv-gpu/bin/python3 tools/concept_rank.py --ckpt "lens/runs/chat/dec_$1_m${MAXW}w_ep$EPOCHS.npz" \
    --windows 100 --fixed-runner --out "lens/runs/DECORR-SUPPLY-$1.json" 2>&1 \
    | grep -E 'positive supply|top-8|P1 verdict|max \|sum' | tee -a "$OUT"
  return 0
}

run_arm ctrl 0.0    || exit 1
run_arm d02  0.2    || exit 1

echo "--- P1/P2 verdict: compare positive supply (P1) and final VAL acc (P2) above ---" | tee -a "$OUT"
echo "DECORR COMPLETE" | tee -a "$OUT"
