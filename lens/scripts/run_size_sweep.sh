#!/bin/bash
# SIZE SWEEP - the gate battery run at EVERY network size, which was missing.
#
# The criticism this answers: the readability gate had only ever been run at d=256, so "readable" was
# never tested across sizes before training moved on. This runs the SAME recipe at d=128, 256 and 512 and
# puts every gate side by side:
#   * gradient check at that width (engine correctness at that size, not just at one size)
#   * honest next-word accuracy (assistant-masked, held-out windows that do not appear in train)
#   * the RANDOM readability gate (false fires, units per window, explain residual, counterfactual)
#   * stop control through the shared decoder, rules off and on
#   * text quality (trigram position between shuffled and real)
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   1. honest accuracy rises with width: d=128 < d=256 < d=512
#   2. the readability gate FAILS at EVERY size -- because the cause is how broadly the concept units
#      fire, not how much capacity the model has; more width will not fix a naming problem
#   3. stop control hands the turn back 0/10 at every size (a bag-of-words concept layer cannot tell
#      where it is in the sequence, and that does not change with width)
#   4. text quality is WORD SALAD at every size
# If any prediction is wrong, the arm's result is reported as a failed prediction, not quietly dropped.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/SIZE-SWEEP.txt
LOG=/tmp/size_sweep.log
exec > "$LOG" 2>&1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"

echo "=== SIZE SWEEP $(date '+%F %H:%M')" | tee -a "$OUT"
echo "recipe held fixed across sizes: 24 epochs, window 32, batch 256, lr 0.4 decay 0.5, lam_concept 0.3," | tee -a "$OUT"
echo "lam_act 0.0, assistant-only masking, --train-turn-end, merged honest corpus, seed 0" | tee -a "$OUT"
echo "predictions: accuracy rises with width; readability FAILS at every size; stop 0/10 at every size" | tee -a "$OUT"

# wait for other GPU work to finish (the readability arms) so two jobs never share the 6 GB card
while pgrep -f "[t]rain_chat" > /dev/null; do sleep 60; done
echo "GPU quiet, starting $(date '+%H:%M')" | tee -a "$OUT"

for D in 128 256 512; do
  NAME="sz_d${D}_24"
  if grep -q "## SIZE $D DONE" "$OUT" 2>/dev/null; then
    echo "## SIZE $D: already finished -- skipping" | tee -a "$OUT"; continue
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## SIZE d=$D   $(date '+%H:%M')" | tee -a "$OUT"

  echo "--- gate 0: gradient check at width $D" | tee -a "$OUT"
  .venv-gpu/bin/python3 -u lens/scripts/gradcheck_width.py --width "$D" 2>&1 | tail -3 | tee -a "$OUT"

  # RESUME: a size that was interrupted continues from its newest checkpoint (d=128 reached ep11 when a
  # machine-wide OOM kill took the process group down on 2026-10-18; that work is not thrown away)
  CK=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
  RES=""
  if [ -n "$CK" ]; then
    DONE=$(basename "$CK" .npz | sed 's/.*_ep//')
    RES="--resume $CK --resume-ep $DONE"
    echo "  resuming $NAME from epoch $DONE of 24" | tee -a "$OUT"
  fi
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$D" --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    $RES --out "$NAME" 2>&1 | tee -a "$OUT.raw"

  SHIP=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $SHIP" | tee -a "$OUT"

  echo "--- gate 1: honest next-word accuracy" | tee -a "$OUT"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ext_honest" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
    | grep -E "assistant-masked|plain pool" | tee -a "$OUT"

  echo "--- gate 2: RANDOM readability gate" | tee -a "$OUT"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest" SHIP="$SHIP" \
    bash lens/scripts/run_random_readability.sh "$SHIP" 2>&1 \
    | grep -E "explain exact|false fires|answer changes|units firing|honest fires|READABILITY GATE" \
    | tee -a "$OUT"

  echo "--- gate 3: stop control (shared decoder)" | tee -a "$OUT"
  SHIP="$SHIP" bash lens/scripts/run_stop_control_v2.sh "$SHIP" 2>&1 \
    | grep -E "handed the turn back|ended on a sentence|unique tokens|stopped by" | tee -a "$OUT"

  echo "--- gate 4: text quality" | tee -a "$OUT"
  .venv-gpu/bin/python3 -u lens/scripts/talk_quality.py "$SHIP" "$NAME" 2>&1 \
    | grep -iE "this model|VERDICT" | tail -3 | tee -a "$OUT"

  echo "## SIZE $D DONE" | tee -a "$OUT"
done

echo "=== SIZE SWEEP DONE $(date '+%F %H:%M')" | tee -a "$OUT"