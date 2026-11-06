#!/bin/bash
# opt_smoke.sh -- the smoke test for PLAN item (d2b), predictions in
# lens/runs/OPT-VS-SGD-PREDICTIONS.txt (P42a, P42b, P42c).
#
# FOUR invocations, all appending to lens/runs/OPT-SMOKE.txt (never overwriting):
#   A  the PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix.py), default step rule   -> control
#   B  the CHANGED script, default step rule (--opt sgd)                                -> P42a: A vs B
#   C  the CHANGED script, --opt relgrad                                                -> P42b (finite, runs)
#   D  the CHANGED script, FULL 12-epoch config + --probe-grads                         -> P42c (the decay
#                                                                                         number, no training)
# The tiny config is --max-windows 20000 --epochs 2 so A/B/C finish in minutes. D runs the REAL
# 12-epoch/1.5M-window configuration and exits before training, which is the only way to print the
# per-step decay factor for 70,320 steps without paying for the run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
OUT=lens/runs/OPT-SMOKE.txt
TINY="--d 256 --window 32 --epochs 2 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
--seed 0 --device gpu --assistant-only --train-turn-end --max-windows 20000"

{
  echo "=== OPT SMOKE START $(date '+%F %H:%M:%S')"
  echo "--- A: PRE-CHANGE script (lens/runs/_tmp/train_chat_prefix.py), default step rule"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/runs/_tmp/train_chat_prefix.py $TINY --out _smoke_A
  echo "A EXIT=$?"
  echo "--- B: CHANGED script, --opt sgd (must be identical to A: P42a)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $TINY --out _smoke_B
  echo "B EXIT=$?"
  echo "--- C: CHANGED script, --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02 (P42b)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $TINY \
    --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02 --out _smoke_C
  echo "C EXIT=$?"
  echo "--- D: CHANGED script, FULL config (12 epochs, 1.5M windows), --probe-grads (P42c, NO training)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02 --probe-grads --out _smoke_D
  echo "D EXIT=$?"
  echo "=== OPT SMOKE DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
