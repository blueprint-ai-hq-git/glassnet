#!/bin/bash
# MEMORISE VERSUS GENERALISE - the third thing asked for, and the last untouched one (2026-11-20 20:25)
#
# THE TRAP DATASET, built in lens/memg.py and generated fresh per seed:
#   RULE rows   two-digit addition. 600 train pairs, 200 HELD-OUT pairs. A memoriser cannot answer held-out pairs; only a model that
#               learned the rule can.
#   FACT rows   a 3-digit key maps to a 2-digit value and EVERY KEY APPEARS EXACTLY ONCE. That mapping cannot be generalised from -
#               it is answerable only by storing it.
#   Same model, same training run, both kinds in the batch. That is what makes the two measurable apart.
#
# THE SWITCH (the decision, which is the actual ask): store a context only when the model is WRONG at it AND it appears EXACTLY ONCE
# in training. A once-seen context has nothing to generalise from, so remembering it is the only correct response; repeated contexts
# are treated as evidence of a rule and left to the weights.
#
# SMOKE-TESTED FIRST (60 steps, tiny model): fact accuracy 12.25% -> 100.00%, rule accuracy unchanged at 22.33%, collision rate 0.00%,
# OUTCOME SWITCH_WORKS. Two bugs found and fixed before this launch: np.stack on variable-length rows (rule 9 tokens, fact 7).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 memorisation: with the switch, fact accuracy is at least 3x the model-only figure.
#   P2 generalisation is not hurt: held-out rule accuracy changes by no more than 1.0 point.
#   P3 no collision: at most 1% of held-out rule contexts match a stored fact context, so P2 cannot be an artefact of the store
#      swallowing rule questions.
# FALSIFIER: held-out rule accuracy falls by 2.0 points or more with the switch on -> this is a TRADE (memorisation bought at the
#   cost of generalisation) and it must be reported as a trade, never as a free win.
#
# SCOPE, STATED UP FRONT SO IT CANNOT BE OVERREAD: this is memorise-vs-generalise at INFERENCE time with an exact external store. It
#   does not change the weights. The weight-integrated version of memory was already falsified in this project by measurement (the
#   readable memory bank: +0.49 against a +1.0 bar, falsifier fired). What is being tested here is the DECISION RULE, not gradient memory.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M)}
OUT=lens/runs/MEMG-${RUN_TAG}.txt
: >> "$OUT"
echo "=== memorise vs generalise, 3 seeds, $(date '+%F %T') ===" | tee -a "$OUT"
for S in 0 1 2; do
  LOG="lens/runs/chain_memg_s${S}_${RUN_TAG}.out"
  echo "--- seed $S ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/memg.py --steps 1500 --batch 32 --d 64 --layers 2 --heads 4 --seed "$S" > "$LOG" 2>&1
  RC=$?
  OUTCOME=$(grep -m1 'OUTCOME:' "$LOG" | sed 's/.*OUTCOME: //')
  if [ -z "$OUTCOME" ]; then echo "  seed $S CRASHED rc=$RC" | tee -a "$OUT"; tail -4 "$LOG" | tee -a "$OUT"; exit 1; fi
  grep -E '^rule accuracy|^fact accuracy|switch hits|^P1 |^P2 |^P3 |FALSIFIER|OUTCOME' "$LOG" | tee -a "$OUT"
done
echo "MEMG COMPLETE" | tee -a "$OUT"
