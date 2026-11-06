#!/bin/bash
# GATE: does training the turn-end signal make the model stop?
#
# PREDICTION, WRITTEN BEFORE THE RUN (project rule: name the bar first, never after):
#   * the stop-control test moves OFF 0/10 -- the model hands the turn back in most generations
#   * the natural answer length is finite (tens of tokens), not the full 150-token budget
#   * honest next-word accuracy does NOT collapse (if it does, the turn-end weight is too strong and
#     the run is reported as a failure, not excused)
# BASELINE being beaten: ship sb_S0_d256_noattn_ep96 -> handed back 0/10, 0/10 ended on punctuation,
# mean unique tokens 58.8% (measured 2026-10-17, saved as lens/runs/stop_control_sb_S0_d256_noattn_ep96.json).
#
# One change only: --train-turn-end added. Everything else matches the S0 recipe (same corpus, same
# size, same seed, same window), except epochs: 24 instead of 96, so the test is cheap. That difference
# is stated here so nobody later reads this as a like-for-like accuracy comparison.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/TURNEND-GATE.txt
LOG=/tmp/turnend_gate.log
exec > "$LOG" 2>&1

echo "=== TURN-END GATE  $(date '+%F %H:%M')" | tee -a "$OUT"
echo "PREDICTION: stop test moves off 0/10; answer length finite; accuracy does not collapse." | tee -a "$OUT"

# RESUME support: an interrupted run continues from its newest checkpoint instead of restarting.
CK=$(ls -t lens/runs/chat/te_T1_turnend_d256_24_ep*.npz 2>/dev/null | head -1)
RES=""
if [ -n "$CK" ]; then
  DONE=$(basename "$CK" .npz | sed 's/.*_ep//')
  RES="--resume $CK --resume-ep $DONE"
  echo "## resuming from epoch $DONE of 24 ($CK)" | tee -a "$OUT"
fi

# NO grep filter here: filtering buffered the output and swallowed the traceback when the run died.
# Full stdout goes to the log; the interesting lines are pulled out afterwards.
CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest" \
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
  $RES --out te_T1_turnend_d256_24 2>&1 | tee -a "$OUT.raw"

SHIP=lens/runs/chat/te_T1_turnend_d256_24_ep24.npz
[ -f "$SHIP" ] || SHIP=$(ls -t lens/runs/chat/te_T1_turnend_d256_24_ep*.npz | head -1)
echo "## SHIP $SHIP" | tee -a "$OUT"

echo "## honest next-word accuracy (must not collapse):" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext_honest" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
  | grep -E "SHIP|RAW|HONEST|assistant-masked" | tee -a "$OUT"

echo "## STOP TEST on the new ship:" | tee -a "$OUT"
CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest" STOP_TEST_SHIP="$PWD/$SHIP" \
  bash lens/scripts/run_stop_control.sh 2>&1 \
  | grep -E "STOP CONTROL|generations:|handed the turn|ended on|unique tokens|when it DOES|VERDICT" \
  | tee -a "$OUT"

echo "## text quality (for context, not the gate):" | tee -a "$OUT"
bash lens/scripts/run_talk_quality.sh "$SHIP" te_T1_turnend_d256_24 "$PWD/data/chat/prepared_ext_honest" 2>&1 \
  | grep -E "trigram|this model|VERDICT" | tee -a "$OUT"

echo "=== GATE DONE $(date '+%F %H:%M')" | tee -a "$OUT"