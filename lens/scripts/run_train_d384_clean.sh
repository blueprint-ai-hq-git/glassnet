#!/bin/bash
# d=384 on the CLEAN corpus, 12 epochs. Epochs are deliberately LOW: d=512 at 48 epochs overfit (honest val
# 24.09% and falling while train reached 34.20%), so this rung tests capacity WITHOUT giving it time to overfit.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== D384 CLEAN TRAIN 12 EPOCHS $(date '+%F %H:%M')"
  echo "prediction: if capacity helps at all without overfitting, this beats the honest 27.24% baseline;"
  echo "if it lands at or below 27.24%, capacity is settled as NOT the limit and the numeric path is the answer."
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 384 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 1500000 --out cl_d384_clean_12
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/D384-CLEAN.txt 2>&1
