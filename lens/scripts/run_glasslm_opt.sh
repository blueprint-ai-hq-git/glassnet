#!/bin/bash
# DOES THE OPTIMISER SET ITS OWN STEP SIZES? (2026-11-20 17:25)
#
# HIS ASK: "when we feed the data it automatically aligns itself... automatic parameters allocation". Today every knob is hand-set:
# one learning rate multiplies every tensor. The adaptive form (added in lens/glass_lm.py) gives each tensor its own measured scale,
# so the global lr becomes dimensionless instead of a physical step size.
#
# THE TEST IS NOT "does it beat a tuned constant" - a tuned constant is already tuned. The test is WHERE THE CONSTANT BREAKS:
# if the lr is set 10x too high, plain momentum should collapse and the adaptive form should not, because it divides out the scale.
#
# ARMS (emb_rank 32 everywhere so the model is the 454,336-param one, seed 0, 600 steps, identical data order):
#   1  momentum  lr 0.05    known-good reference (12.51% at rank 32)
#   2  momentum  lr 0.50    10x too high
#   3  adaptive  lr 0.50    the automation
#   4  adaptive  lr 5.00    another 10x, to see whether the knob really stops mattering
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 plain momentum at lr 0.50 collapses: at least 2.0 points below arm 1.
#   P2 adaptive at lr 0.50 lands within 1.0 point of arm 1 -> the lr knob stops mattering once the scale is measured per tensor.
#   P3 adaptive's spread across lr 0.50 and 5.00 is smaller than plain momentum's spread across 0.05 and 0.50.
# FALSIFIER: adaptive at lr 0.50 is also at least 2.0 points below arm 1 -> per-tensor normalisation does NOT remove lr sensitivity,
#   and ONE variant (per-element RMS) gets tried before this direction is judged. Not a sweep of learning rates.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
# RUN-TAGGED OUTPUT (2026-11-20 17:42). The fixed name plus `: >` truncated the previous run's summary, which had just recorded
# P1 (a diverged arm). Same family as the checkpoint-tag clobber: a fixed output name means every run destroys the last one's record.
# The summary now carries a timestamp tag and is APPENDED to, and each arm's log carries the same tag, so no run can overwrite another.
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/GLASSLM-OPT-${RUN_TAG}.txt
: >> "$OUT"
echo "=== does the optimiser set its own step sizes? $(date '+%F %T') ===" | tee -a "$OUT"
run_arm() {   # label opt lr
  echo "--- $1 (opt=$2 lr=$3) ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/glass_lm.py \
    --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 \
    --emb-rank 32 --seed 0 --opt "$2" --lr "$3" --log-every 300 \
    >> "lens/runs/chain_glopt_${1}_${RUN_TAG}.out" 2>&1
  RC=$?
  echo "  rc=$RC" | tee -a "$OUT"
  # RUNNER FIX (2026-11-20 17:40): exit 3 means the arm DIVERGED, which is the evidence this test exists to collect. The first
  # version aborted the whole run on it and threw away arms 3 and 4 - the automation being tested.
  if [ "$RC" -eq 3 ]; then
    echo "  $1 DIVERGED (recorded as a result; continuing)" | tee -a "$OUT"
    return 0
  fi
  [ "$RC" -ne 0 ] && { echo "  ARM $1 FAILED rc=$RC" | tee -a "$OUT"; return 1; }
  grep -E 'optimiser:|HELD-OUT|DIVERGED' "lens/runs/chain_glopt_${1}_${RUN_TAG}.out" | tail -2 | tee -a "$OUT"
  return 0
}
run_arm mom_lr005   momentum 0.05 || exit 1
run_arm mom_lr050   momentum 0.50 || exit 1
run_arm ada_lr050   adaptive 0.50 || exit 1
run_arm ada_lr500   adaptive 5.00 || exit 1
echo "--- P1/P2/P3: compare the four HELD-OUT lines above ---" | tee -a "$OUT"
echo "GLASSLM-OPT COMPLETE" | tee -a "$OUT"
