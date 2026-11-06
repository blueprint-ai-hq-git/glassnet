#!/bin/bash
# CONTROL-TOKEN LADDER: small -> big, tested with RANDOM draws at every size.
#
# What the user asked for: use the standard control tokens (<|im_start|> / <|im_end|>), then train and
# test SMALL TO BIG, with random sampling at every rung, so the claims are "tested in every condition"
# rather than demonstrated once. Every gate below draws its windows/prompts at random with a recorded
# seed, so anyone can reproduce or contradict the number.
#
# Sizes: d = 64, 128, 256, 512, 1024. The parameter counts stay inside the user's <=20M target (d=1024 is
# about 13M). Recipe is held fixed across the ladder; ONLY the width changes, so a difference between
# rungs is attributable to width.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 honest next-word accuracy rises monotonically with width (64 < 128 < 256 < 512 <= 1024)
#   P2 the readability gate FAILS at every rung: false fires stay in the hundreds and the counterfactual
#      check stays ~200/200. Width does not fix a naming problem.
#   P3 P(<|im_end|>) at true turn boundaries rises above zero at EVERY rung (that is what the dedicated
#      token is for), but "handed the turn back" stays low while the text is still word salad
#   P4 the number of concepts that fire per window grows with width (more capacity, broader firing)
# A failed prediction is reported as failed, in the ledger, with the numbers that failed it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ctrl"
OUT=lens/runs/LADDER-CTRL.txt
LOG=/tmp/ladder_ctrl.log
exec > "$LOG" 2>&1

echo "=== CONTROL-TOKEN LADDER (small -> big, random draws) $(date '+%F %H:%M')" | tee -a "$OUT"
echo "corpus data/chat/prepared_ctrl  vocab 12,002  format <|im_start|> role <text> <|im_end|>" | tee -a "$OUT"
echo "recipe fixed: 24 epochs, window 32, batch 256, lr 0.4 decay 0.5, lam_concept 0.3, lam_act 0.0," | tee -a "$OUT"
echo "assistant-only masking, --train-turn-end, seed 0" | tee -a "$OUT"
echo "predictions: accuracy rises with width; readability FAILS at every rung; P(<|im_end|>) > 0 at every" | tee -a "$OUT"
echo "rung; units firing per window grows with width" | tee -a "$OUT"
echo "random seeds used by the gates: readability 20261018 / 7 / 12345, boundary probe 20261018 / 4242" | tee -a "$OUT"

while pgrep -f "[t]rain_chat|[r]un_size_sweep" > /dev/null; do sleep 60; done
echo "GPU quiet, starting $(date '+%H:%M')" | tee -a "$OUT"

for D in 64 128 256 512 1024; do
  NAME="ld_d${D}_ctrl_24"
  if grep -q "## RUNG $D DONE" "$OUT" 2>/dev/null; then
    echo "## RUNG $D: already finished -- skipping" | tee -a "$OUT"; continue
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## RUNG d=$D   $(date '+%H:%M')" | tee -a "$OUT"

  echo "--- gate 0: numerical gradient check at width $D (engine correctness at this size)" | tee -a "$OUT"
  .venv-gpu/bin/python3 -u lens/scripts/gradcheck_width.py --width "$D" 2>&1 | tail -2 | tee -a "$OUT"

  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$D" --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --out "$NAME" 2>&1 | tee -a "$OUT.raw"

  SHIP=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
  if [ -z "$SHIP" ]; then echo "## RUNG $D: NO CHECKPOINT -- training failed, stopping the ladder" | tee -a "$OUT"; break; fi
  echo "## SHIP $SHIP" | tee -a "$OUT"
  PARAMS=$(.venv-gpu/bin/python3 -c "
import sys, os
sys.path.insert(0, '$PWD'); sys.path.insert(0, '$PWD/lens'); sys.path.insert(0, '$PWD/lens/scripts')
os.environ['CHAT_DATA_DIR'] = '$PWD/data/chat/prepared_ctrl'
import lens
n = lens.load('$SHIP', device='cpu')
print(f\"{sum(p.data.size for p in n.parameters()):,} params, {n.K} concepts\")" 2>/dev/null)
  echo "## SIZE d=$D: $PARAMS" | tee -a "$OUT"

  echo "--- gate 1: honest next-word accuracy (held-out windows absent from train)" | tee -a "$OUT"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ctrl" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
    | grep -E "assistant-masked|plain pool|WARNING" | tee -a "$OUT"

  echo "--- gate 2: does the model predict the stop token where a turn really ends?" | tee -a "$OUT"
  BOUNDARY_TOKEN="<|im_end|>" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/turnend_probe.py \
    "$SHIP" 2>&1 | grep -E "boundary positions|mean P|top-1|share above|VERDICT" | tee -a "$OUT"

  echo "--- gate 3: stop control through the shared decoder" | tee -a "$OUT"
  SHIP="$SHIP" bash lens/scripts/run_stop_control_v2.sh "$SHIP" 2>&1 \
    | grep -E "handed the turn back|ended on a sentence|unique tokens|stopped by|rule-forced" | tee -a "$OUT"

  echo "--- gate 4: RANDOM readability gate, three independent draws" | tee -a "$OUT"
  for S in 20261018 7 12345; do
    echo "  (random draw seed $S)" | tee -a "$OUT"
    SHIP="$SHIP" N=200 SEED="$S" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
      lens/scripts/random_readability.py 2>&1 \
      | grep -E "false fires|answer changes|units firing|honest fires|READABILITY GATE" | tee -a "$OUT"
  done

  echo "--- gate 5: text quality" | tee -a "$OUT"
  .venv-gpu/bin/python3 -u lens/scripts/talk_quality.py "$SHIP" "$NAME" 2>&1 \
    | grep -iE "this model|VERDICT" | tail -3 | tee -a "$OUT"

  echo "## RUNG $D DONE" | tee -a "$OUT"
done

echo "=== LADDER DONE $(date '+%F %H:%M')" | tee -a "$OUT"