#!/bin/bash
# run_opt_vs_sgd.sh <arm> <seed> -- ONE ARM of the same-seed comparison of STEP RULES on the honest val.
#
# PLAN item (d2b). Predictions P42a-i were written BEFORE any of these runs, in
# lens/runs/OPT-VS-SGD-PREDICTIONS.txt. Nothing about the bars is decided after the numbers exist.
#
# ONE arm = ONE train_chat.py invocation. Deliberately not a loop over arms: the watchdog starts the next
# queue entry only when `pgrep -cf "[t]rain_chat"` sees nothing running, so two arms inside one script
# could overlap with the next queue entry and double the VRAM use.
#
# ARMS (only the step rule differs; corpus, seed, epochs, batch, mask and size are identical):
#   sgd04       plain SGD, lr 0.4, linear decay to 0.2  -- the SHIPPED clean recipe (run_clean_train.sh)
#   sgd10       plain SGD, lr 1.0, linear decay to 0.5  -- does plain SGD survive a bigger step here?
#   relgrad005  OUR rule (lens/opt.py), opt_lr 0.005, decay to 0.02 at the last step
#   relgrad05   OUR rule, opt_lr 0.05 (the median value the gradient probe matched to SGD@0.4)
#   relgrad5    OUR rule, opt_lr 0.5
#   relgrad05c  OUR rule, opt_lr 0.05, CONSTANT step (--opt-decay-final 1.0) -- the P42g control
# The relgrad step sizes span 100x around the probe's matched value 0.0441256 (lens/runs/OPT-GRAD-SCALE.txt).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
ARM="${1:?usage: run_opt_vs_sgd.sh <arm> <seed>}"
SEED="${2:?usage: run_opt_vs_sgd.sh <arm> <seed>}"
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
EPOCHS="${OPT_EPOCHS:-12}"
OUTF=lens/runs/OPT-VS-SGD.txt

case "$ARM" in
  sgd04)      OPTARGS="--opt sgd --lr 0.4 --lr-decay 0.5" ;;
  sgd10)      OPTARGS="--opt sgd --lr 1.0 --lr-decay 0.5" ;;
  relgrad005) OPTARGS="--opt relgrad --opt-lr 0.005 --opt-decay-final 0.02" ;;
  relgrad05)  OPTARGS="--opt relgrad --opt-lr 0.05  --opt-decay-final 0.02" ;;
  relgrad5)   OPTARGS="--opt relgrad --opt-lr 0.5   --opt-decay-final 0.02" ;;
  relgrad05c) OPTARGS="--opt relgrad --opt-lr 0.05  --opt-decay-final 1.0" ;;
  *) echo "=== OPT-VS-SGD unknown arm '$ARM' -- nothing run $(date '+%F %H:%M:%S')" >> "$OUTF"; exit 2 ;;
esac
TAG="opt_${ARM}_s${SEED}"

{
  echo "=== OPT-VS-SGD ARM $ARM SEED $SEED EPOCHS $EPOCHS  start $(date '+%F %H:%M:%S')"
  echo "    corpus data/chat/prepared_mix_ctrl_honest (VERDICT CLEAN: val disjoint from train at the"
  echo "    training window), 1,500,000 windows; recipe d=256 W=32 batch 256 assistant-only"
  echo "    train-turn-end, --max-gap 0 (OFF: this comparison is about the step rule, P42i)"
  echo "    args: $OPTARGS"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs "$EPOCHS" --batch 256 --lam-concept 0.3 \
    --seed "$SEED" --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    $OPTARGS --out "$TAG"
  echo "ARM EXIT=$?"
  echo "=== OPT-VS-SGD ARM $ARM SEED $SEED done $(date '+%F %H:%M:%S')"
  echo "    per-epoch rows: lens/runs/chat/${TAG}_history.jsonl   checkpoints: lens/runs/chat/${TAG}_ep*.npz"
} >> "$OUTF" 2>&1
