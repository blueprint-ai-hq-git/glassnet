#!/bin/bash
# SMALL -> BIG QUEUE (Yahya's order, 2026-10-16): fix the SMALL model first, then the BIG one.
#
# PRE-REGISTERED BARS (written before any arm runs; do not move them afterwards)
#  1. Accuracy: clean-val assistant-masked next-token accuracy on the MERGED val split
#     (corpus prepared_ext, vocab md5 01b1eba6f8, n=29,515 windows). Every arm is trained and
#     measured on that one family, so the numbers are comparable to each other.
#  2. Talk quality: the v2 trigram meter (bar taken from a shuffled-word control built out of real
#     held-out text). Output must land ABOVE the midpoint between shuffled and real.
#  3. The attention lever counts as an improvement ONLY if it raises clean-val accuracy AND moves
#     talk quality toward real text. Either alone is reported, not claimed as a win.
#
# ARMS (one change at a time: width, then attention; seed fixed at 0)
#   S0  d256  no attention   <- the SMALL model's baseline on the current corpus with the fixed code
#   S1  d256  attention      <- the token-order lever, small model
#   B0  d512  no attention   <- the BIG model, FRESH (also settles why resumed M0 collapsed)
#   B1  d512  attention      <- the token-order lever, big model
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext"
OUT=lens/runs/SMALL-BIG.txt
LOG=/tmp/small_big.log
exec >> "$LOG" 2>&1
echo "=== SMALL->BIG QUEUE START $(date '+%F %H:%M')"

run_arm () {
  local name="$1" d="$2" extra="$3"
  echo "############################################################" | tee -a "$OUT"
  echo "## ARM $name   d=$d   extra='${extra:-none}'   $(date '+%H:%M')" | tee -a "$OUT"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$d" --window 32 --epochs 96 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    $extra --out "$name" 2>&1 | grep -E "window cap|windows |ep 1/96|ep 24/96|ep 48/96|ep 72/96|ep 96/96|final" \
    | tee -a "$OUT"
  local ship="lens/runs/chat/${name}_ep96.npz"
  [ -f "$ship" ] || ship=$(ls -t lens/runs/chat/${name}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $name -> $ship" | tee -a "$OUT"
  echo "## (A) merged clean val:" | tee -a "$OUT"
  EVAL_DATA_DIR="$CHAT_DATA_DIR" bash lens/scripts/run_eval_clean.sh "$ship" 2>&1 \
    | grep -E "assistant-masked|plain pool|SHIP" | tee -a "$OUT"
  echo "## (B) talk quality (v2 meter):" | tee -a "$OUT"
  bash lens/scripts/run_talk_quality.sh "$ship" "$name" "$CHAT_DATA_DIR" 2>&1 \
    | grep -E "trigram log-prob|this model|VERDICT" | tee -a "$OUT"
}

run_arm  "sb_S0_d256_noattn" 256 ""
run_arm  "sb_S1_d256_attn"   256 "--attention"
run_arm  "sb_B0_d512_noattn" 512 ""
run_arm  "sb_B1_d512_attn"   512 "--attention"

echo "=== SMALL->BIG QUEUE DONE $(date '+%F %H:%M')" | tee -a "$OUT"
