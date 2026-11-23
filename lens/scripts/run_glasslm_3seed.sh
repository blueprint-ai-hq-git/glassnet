#!/bin/bash
# THREE-SEED CONFIRMATION: factored rank 32 vs full table (2026-11-20 16:35)
#
# WHY THIS RUN EXISTS AND NOTHING ELSE: rank 32 measured 12.51% against the full table's 12.33%, i.e. +0.18 on ONE seed. The project
# rule is that a claim needs 3+ seeds, and +0.18 is inside the noise this project has already measured elsewhere (sd 0.93). So the
# claim currently supported is "equal within noise at 1.84x fewer params" - this run tests whether even THAT survives, and whether
# the +0.18 is real or a lucky draw.
#
# PAIRED ON SEED, as the project requires: seed s runs BOTH arms, so the comparison is a paired delta, not two means from different
# data orders.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 paired mean delta (factored r=32 minus full table) is within -1.0 and +1.0 points -> the two are equal within noise, and
#      "1.84x fewer parameters at equal accuracy" is confirmed at 3 seeds.
#   P2 all three full-table seeds land in 11.3-13.5% and all three factored seeds in 11.3-13.5% (no seed collapses), so the
#      comparison is not an artefact of one bad run.
# FALSIFIER: paired mean delta worse than -1.0 -> factored embeddings LOSE accuracy at rank 32, the +0.18 was a lucky draw, and the
#   publishable claim becomes "1.84x smaller but more than a point worse", not "equal".
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/GLASSLM-3SEED.txt
: > "$OUT"
echo "=== factored r=32 vs full table, 3 seeds paired, $(date '+%F %T') ===" | tee -a "$OUT"
for S in 0 1 2; do
  for RANK in 0 32; do
    TAG="s${S}_r${RANK}"
    echo "--- seed $S emb_rank=$RANK ---" | tee -a "$OUT"
    env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
      ${PYTHON:-python3} -u lens/glass_lm.py \
      --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 --lr 0.05 \
      --seed "$S" --emb-rank "$RANK" --log-every 300 >> "lens/runs/chain_gl3_$TAG.out" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "  $TAG FAILED rc=$RC" | tee -a "$OUT"; exit 1; fi
    grep -E 'HELD-OUT|model:' "lens/runs/chain_gl3_$TAG.out" | tail -2 | tee -a "$OUT"
  done
done
echo "--- paired deltas (factored minus full), per seed ---" | tee -a "$OUT"
for S in 0 1 2; do
  A=$(grep -oE 'HELD-OUT  loss [0-9.]+  acc [0-9.]+' "lens/runs/chain_gl3_s${S}_r32.out" | tail -1 | grep -oE '[0-9.]+$')
  B=$(grep -oE 'HELD-OUT  loss [0-9.]+  acc [0-9.]+' "lens/runs/chain_gl3_s${S}_r0.out"  | tail -1 | grep -oE '[0-9.]+$')
  echo "  seed $S: factored $A  full $B  delta $(awk -v a=$A -v b=$B 'BEGIN{printf "%+.2f", a-b}')" | tee -a "$OUT"
done
echo "GLASSLM-3SEED COMPLETE" | tee -a "$OUT"
