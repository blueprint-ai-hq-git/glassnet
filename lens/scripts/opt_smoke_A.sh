#!/bin/bash
# opt_smoke_A.sh -- RE-RUN of arm A of the smoke (the PRE-CHANGE train_chat.py, lens/runs/_tmp/).
#
# WHY THE RE-RUN: the first attempt (17:06, A EXIT=1) died with
#   ModuleNotFoundError: No module named 'chat_concepts'
# because a script run by PATH gets ITS OWN directory on sys.path, and the pre-change copy lives in
# lens/runs/_tmp/ while chat_concepts.py lives in lens/scripts/. That is a harness fault, not a model
# fault, and it is fixed by PYTHONPATH instead of by moving the copy (the copy must stay byte-identical
# to what was on disk before the edit - md5 87de1332446b7a654947fdc87ae54ae8, verified at copy time).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
OUT=lens/runs/OPT-SMOKE.txt
TINY="--d 256 --window 32 --epochs 2 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
--seed 0 --device gpu --assistant-only --train-turn-end --max-windows 20000"
{
  echo "=== OPT SMOKE A RE-RUN (PYTHONPATH=$PYTHONPATH) $(date '+%F %H:%M:%S')"
  echo "--- A2: PRE-CHANGE script, default step rule (the P42a control)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/runs/_tmp/train_chat_prefix.py $TINY --out _smoke_A
  echo "A2 EXIT=$?"
  echo "=== OPT SMOKE A DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
