#!/bin/bash
# opt_smoke_AC.sh -- the two smoke arms that had to be re-run:
#   A2  the PRE-CHANGE train_chat.py (the P42a control) with PYTHONPATH set, because a script run by
#       PATH gets its own directory on sys.path and the copy lives outside lens/scripts/
#   C2  --opt relgrad on the GPU, after the lens/opt.py device fix (the first C died with
#       "TypeError: Implicit conversion to a NumPy array is not allowed" from np.asarray on a CUPY array)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
OUT=lens/runs/OPT-SMOKE.txt
TINY="--d 256 --window 32 --epochs 2 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
--seed 0 --device gpu --assistant-only --train-turn-end --max-windows 20000"
{
  echo "=== OPT SMOKE A2+C2 RE-RUN $(date '+%F %H:%M:%S')"
  echo "--- A2: PRE-CHANGE script, default step rule, PYTHONPATH=lens/scripts (the P42a control)"
  PYTHONPATH="$PWD/lens/scripts" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/runs/_tmp/train_chat_prefix.py $TINY --out _smoke_A
  echo "A2 EXIT=$?"
  echo "--- C2: CHANGED script, --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02, AFTER the device fix"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $TINY \
    --opt relgrad --opt-lr 0.05 --opt-decay-final 0.02 --out _smoke_C2
  echo "C2 EXIT=$?"
  echo "=== OPT SMOKE A2+C2 DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
