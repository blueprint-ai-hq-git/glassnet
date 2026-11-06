#!/bin/bash
# STANDARD CONTROL TOKENS (ChatML) - train and judge.
#
# THE CHANGE (one change): the model gets the control tokens every mainstream chat model ships,
# <|im_start|> and <|im_end|>, instead of borrowing the ordinary word "human" as its stop signal.
# Corpus: data/chat/prepared_ctrl, 38,997,354 tokens, 167,772 <|im_end|> stop tokens at real turn ends,
# verified by expanding the control tokens back to the original stream (exact match).
#
# WHY: measured on the old format, at 71 true turn boundaries the model gave the borrowed word "human"
# probability 0.0000 and never ranked it top-1 (0/71). "human" is an ordinary word (1.49% of tokens) and
# the marker was TWO tokens, so the stop decision was split and ambiguous.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   1. P(<|im_end|>) at true turn boundaries rises well above zero -- the tag problem is fixed by a
#      dedicated token (this is the whole point of the change)
#   2. the model emits <|im_end|> on its own in some generations, so stop-control's "handed the turn
#      back" moves off 0/10
#   3. honest next-word accuracy does not collapse relative to the same recipe without control tokens
#      (the text content is unchanged; only the markers differ)
#   4. the readability gate is unchanged -- control tokens say nothing about how broadly concept units
#      fire, so it should still FAIL until the position/sparsity work lands. Predicting no change there.
# If prediction 1 or 2 fails, that is reported as a failed prediction, and the honest conclusion is that
# the stop signal needs more than tags (the concept layer still cannot see WHERE it is in the sequence).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ctrl"
OUT=lens/runs/CTRL-TOKENS.txt
LOG=/tmp/ctrl_tokens_run.log
exec > "$LOG" 2>&1

echo "=== STANDARD CONTROL TOKENS (ChatML) $(date '+%F %H:%M')" | tee -a "$OUT"
echo "corpus data/chat/prepared_ctrl  vocab 12,002  <|im_start|>=12000 <|im_end|>=12001" | tee -a "$OUT"
echo "predictions: (1) P(<|im_end|>) at boundaries rises above zero (2) some generations emit it" | tee -a "$OUT"
echo "             (3) accuracy does not collapse (4) readability gate unchanged, still FAIL" | tee -a "$OUT"

# wait for other GPU work (the size sweep) so two jobs never share the 6 GB card
while pgrep -f "[t]rain_chat|[r]un_size_sweep" > /dev/null; do sleep 60; done
echo "GPU quiet, starting $(date '+%H:%M')" | tee -a "$OUT"

NAME=ct_T1_chatml_d256_24
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
  --out "$NAME" 2>&1 | tee -a "$OUT.raw"

SHIP=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
echo "## SHIP $SHIP" | tee -a "$OUT"

echo "--- PREDICTION 1: does the model predict the stop token where a turn really ends?" | tee -a "$OUT"
BOUNDARY_TOKEN="<|im_end|>" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/turnend_probe.py \
  "$SHIP" 2>&1 | grep -E "boundary token|boundary positions|mean P|top-1|share above|prefers|VERDICT" \
  | tee -a "$OUT"

echo "--- PREDICTION 2: stop control through the shared decoder" | tee -a "$OUT"
SHIP="$SHIP" bash lens/scripts/run_stop_control_v2.sh "$SHIP" 2>&1 \
  | grep -E "handed the turn back|ended on a sentence|unique tokens|stopped by|rule-forced" | tee -a "$OUT"

echo "--- PREDICTION 3: honest next-word accuracy" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ctrl" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
  | grep -E "assistant-masked|plain pool|WARNING" | tee -a "$OUT"

echo "--- PREDICTION 4: random readability gate (expected unchanged)" | tee -a "$OUT"
SHIP="$SHIP" bash lens/scripts/run_random_readability.sh "$SHIP" 2>&1 \
  | grep -E "explain exact|false fires|answer changes|units firing|honest fires|READABILITY GATE" \
  | tee -a "$OUT"

echo "=== DONE $(date '+%F %H:%M')" | tee -a "$OUT"