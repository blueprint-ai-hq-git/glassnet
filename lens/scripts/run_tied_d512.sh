#!/bin/bash
# CRITICAL PATH 2 OF 2: THE CAPACITY TEST.
# Same recipe as critical path 1, but 4,000,000 windows instead of 6,000,000: d=512 makes every activation
# twice as wide, and the d=256 attempt at 16.9M windows was already measured at 9.0 GB and climbing on a 14 GB
# machine. Memory is the real budget here, so the window count is what gets traded, not the comparison.
# Same data and recipe as critical path 1, but d=512 (~20M parameters, still inside the 50M cap). This is the
# honest test of "is the model just too small", which is the explanation to try if more DATA does not move
# held-out. Nothing else changes, so if this moves and the data run did not, capacity was the constraint.
#
# PREDICTIONS, written BEFORE the run:
#  1. if capacity is the constraint, held-out beats the d=256 full-corpus run.
#  2. if it does not beat it, then neither data volume nor capacity explains the ceiling, and the next suspect
#     is the concept/answer interface itself (how many concepts survive, and what the concept loss is doing).
#  3. the talk test decides, in every case.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS d=512 4 epochs $(date '+%F %H:%M')"
  echo "reference: d=256 tied full corpus (see TIED-ALLDATA.txt)"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 512 --window 32 --epochs 4 --batch 128 --lr 0.3 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 4000000 --out ct_d512_alldata_4
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/TIED-D512.txt 2>&1
