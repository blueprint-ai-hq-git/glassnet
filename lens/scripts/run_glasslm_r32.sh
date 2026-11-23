#!/bin/bash
# THE ONE RETRY OWED: factored rank 32 (2026-11-20 16:25)
#
# WHY THIS EXACTLY AND NOTHING ELSE: rank 16 measured 10.86% against the full table's 12.33%. The pre-registered falsifier was
# "below 10.32%", and 10.86% cleared it, so the rule written before that run allows ONE retry at rank 32. Not a sweep, not a new
# mechanism - this is the retry the rule owes.
#
# WHAT IT CANNOT DO, STATED BEFORE THE RUN: rank 32 raises the embedding to 386,112 params, i.e. about 85% of a 454,336-param model.
# So it can rescue the accuracy question but it moves the capacity share the WRONG way for the real goal.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 rank-32 held-out accuracy >= 11.33% (within 1.0 point of the full table's 12.33%).
#   P2 rank-32 total parameters <= 460,000, still >= 1.8x smaller than the full table's 836,352.
# FALSIFIER: rank-32 also below 11.33% -> factored embeddings cost accuracy at every rank tested, this mechanism fails, and the next
#   work is a DIFFERENT mechanism (capacity allocation / smaller vocabulary / character-level), not another rank.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/GLASSLM-R32.txt
: > "$OUT"
echo "=== the owed retry: factored rank 32, $(date '+%F %T') ===" | tee -a "$OUT"
env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
  ${PYTHON:-python3} -u lens/glass_lm.py \
  --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 --lr 0.05 \
  --emb-rank 32 --log-every 150 >> lens/runs/chain_glasslm_r32.out 2>&1
RC=$?
echo "  rc=$RC" | tee -a "$OUT"
[ "$RC" -ne 0 ] && { echo "  FAILED" | tee -a "$OUT"; exit 1; }
grep -E 'model:|OUTPUT-TABLE SHARE|matched constant|HELD-OUT' lens/runs/chain_glasslm_r32.out | tail -4 | tee -a "$OUT"
echo "GLASSLM-R32 COMPLETE" | tee -a "$OUT"
