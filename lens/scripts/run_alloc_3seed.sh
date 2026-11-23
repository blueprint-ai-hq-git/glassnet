#!/bin/bash
# THREE-SEED CONFIRMATION OF CAPACITY ALLOCATION (2026-11-20 19:15)
#
# MEASURED DIRECTION (1 seed, 300 steps): C r16/d96/L2 = 10.87% (+0.58), B r8/d96/L3 = 10.74% (+0.45, and 29% smaller than the default),
# A default r32/d64/L2 = 10.29%. This project has measured paired-delta sd 0.35, so both figures sit inside one standard deviation.
# The claim needs 3 paired seeds - that is what this run is, and it is the only thing this run is.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 paired mean delta (C - A) >= +0.5 -> allocation genuinely buys accuracy at this budget.
#   P2 paired mean delta (B - A) >= 0.0  -> a config with a 30%-table and 29% fewer params is at least equal to the default, which is the
#      capacity finding that matters.
#   P3 every seed of every arm beats the 4.08% matched constant (the arms trained).
# FALSIFIER: paired mean delta (C - A) < +0.2 -> the +0.58 was noise, allocation wins nothing at 300 steps, and this line CLOSES at this
#   budget (no 600-step rescue attempt, no grid widening). The honest sentence then becomes "allocation made no difference here".
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/ALLOC-3SEED-${RUN_TAG}.txt
: >> "$OUT"
echo "=== capacity allocation, 3 paired seeds, $(date '+%F %T') ===" | tee -a "$OUT"
run_cfg() {   # label seed rank d layers
  local LOG="lens/runs/chain_alloc3_${1}_s${2}_${RUN_TAG}.out"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    ${PYTHON:-python3} -u lens/glass_lm.py \
    --steps 300 --batch 16 --T 32 --d "$4" --layers "$5" --heads 4 --d-mlp "$((2*$4))" --windows 20000 \
    --emb-rank "$3" --seed "$2" --opt momentum --lr 0.05 --log-every 300 > "$LOG" 2>&1
  local RC=$?
  local OUTCOME
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then echo "  $1 seed $2 CRASHED rc=$RC" | tee -a "$OUT"; return 1; fi
  local ACC
  ACC=$(grep -oE 'HELD-OUT  loss [0-9.]+  acc [0-9.]+' "$LOG" | tail -1 | grep -oE '[0-9.]+$')
  echo "  $1 seed $2: acc $ACC  ($OUTCOME)" | tee -a "$OUT"
  return 0
}
for S in 0 1 2; do
  echo "--- seed $S ---" | tee -a "$OUT"
  run_cfg A  $S 32 64  2 || exit 1
  run_cfg B  $S  8 96  3 || exit 1
  run_cfg C  $S 16 96  2 || exit 1
done
echo "--- paired deltas per seed ---" | tee -a "$OUT"
for S in 0 1 2; do
  A=$(grep -oE 'A +seed '"$S"': acc [0-9.]+' lens/runs/ALLOC-3SEED-${RUN_TAG}.txt | grep -oE '[0-9.]+$')
  B=$(grep -oE 'B +seed '"$S"': acc [0-9.]+' lens/runs/ALLOC-3SEED-${RUN_TAG}.txt | grep -oE '[0-9.]+$')
  C=$(grep -oE 'C +seed '"$S"': acc [0-9.]+' lens/runs/ALLOC-3SEED-${RUN_TAG}.txt | grep -oE '[0-9.]+$')
  echo "  seed $S: A $A  B $B  C $C   C-A $(awk -v c=$C -v a=$A 'BEGIN{printf "%+.2f", c-a}')   B-A $(awk -v b=$B -v a=$A 'BEGIN{printf "%+.2f", b-a}')" | tee -a "$OUT"
done
echo "ALLOC-3SEED COMPLETE" | tee -a "$OUT"
