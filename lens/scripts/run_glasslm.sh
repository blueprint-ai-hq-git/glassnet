#!/bin/bash
# FIRST REAL TRAINING RUN OF OUR OWN BLOCK (2026-11-20 15:30)
#
# WHAT IS CERTIFIED BEFORE THIS RUNS:
#   forward: attention+residual vs a HAND reference 4.441e-16 | KV-cache equivalence 1.192e-07 | LayerNorm vs hand 0.000e+00
#   backward: every named gradient vs finite differences, worst relative error 8.815e-07, input gradient 2.650e-09
# So this is the first run where nothing underneath it is a guess.
#
# THE BASELINE IS BUILT THE SAME WAY AS THE REST OF THE PROJECT: most frequent TRAIN target after excluding <unk>, scored on the
# same <unk>-excluded held-out targets. (The first version of this file picked token 0 and reported a 0.00% baseline, which made
# "beats constant" worthless - fixed before this run.)
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 held-out accuracy beats the matched constant by at least +1.0 point.
#   P2 loss falls across the run without diverging (no increase of more than 1.0 between consecutive logged checkpoints).
# FALSIFIER: held-out accuracy at or below the matched constant -> the block does not learn on real text, and NOTHING further is
#   built on it until the cause is found. No architecture work, no allocation work, no memorise/generalise work.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
export PYTHONPATH=.
OUT=lens/runs/GLASSLM.txt
: > "$OUT"
echo "=== first real training run of our own block, $(date '+%F %T') ===" | tee -a "$OUT"
env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
  .venv-gpu/bin/python3 -u lens/glass_lm.py \
  --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 --lr 0.05 --log-every 50 \
  2>&1 | tee -a "$OUT"
RC=${PIPESTATUS[0]}
echo "  trainer rc=$RC" | tee -a "$OUT"
echo "GLASSLM COMPLETE" | tee -a "$OUT"
exit "$RC"
