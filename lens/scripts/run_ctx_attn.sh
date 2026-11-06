#!/bin/bash
# WHERE WE ARE STUCK: at 20k windows held-out is pinned at ~11% regardless of width (32/64/128/256 all 9-12%)
# and regardless of passes (3/12/30 all 8-11%). The only lever that ever raised it was data volume. So the two
# remaining architectural suspects are tested here, one change per arm against the SAME 20k reference:
#   A --attention        : token ORDER (without it the model sees a bag of words; word salad is the known symptom)
#   B --window 64        : twice the tokens per example (640k -> 1.28M tokens at the same 20k windows)
#   reference (measured): window 32, no attention -> held-out 10.08 / 11.96, mean 11.02%
# PREDICTIONS, before any arm:
#  1. if disorder is the wall, A beats 11.02% clearly. If A also collapses train accuracy, the lever costs more
#     than it gives - reported either way.
#  2. if the wall is tokens-per-example, B beats 11.02%. If B is flat while token count doubled, the wall is
#     distinct examples, not tokens - and then no flag can fix it; the architecture must.
#  3. any arm that wins gets A THIRD SEED before it is called a win.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; SEED=$2; shift 2
  echo "===== $NAME seed $SEED : $*  $(date '+%H:%M')" >> lens/runs/CTX-ATTN.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 \
    --out "ca_${NAME}_s$SEED" --method ours --opt-lr 1.0 "$@" >> lens/runs/CTX-ATTN.txt 2>&1
  echo "  $NAME seed $SEED exit=$?" >> lens/runs/CTX-ATTN.txt
}
run attn_w32 0 --window 32 --attention
run attn_w32 1 --window 32 --attention
run w64 0 --window 64
run w64 1 --window 64
echo "=== CTX-ATTN DONE $(date '+%F %H:%M')" >> lens/runs/CTX-ATTN.txt
