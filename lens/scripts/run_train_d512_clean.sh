#!/bin/bash
# Next rung on the CLEAN corpus: d=512, 48 epochs. Queued so the machine is never idle after this finishes.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== D512 CLEAN TRAIN $(date '+%F %H:%M')"
  echo "corpus: prepared_mix_ctrl_honest (294,225,090 tokens, val disjoint from train)"
  echo "prediction: higher capacity should raise the honest masked accuracy above the 27.35% measured at d=256;"
  echo "if it does not, capacity is not the limit and the numeric-path change is the answer."
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 512 --window 32 --epochs 48 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 1500000 --out cl_d512_clean_48
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/D512-CLEAN.txt 2>&1
