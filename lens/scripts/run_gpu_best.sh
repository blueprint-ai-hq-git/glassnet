#!/bin/bash
# batch 1024 = the accuracy setting (35.26% baseline family), so GPU is busy with work worth doing.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 1024 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 1 --device gpu --assistant-only \
  --train-turn-end --direct --direct-mode tied --out gpu_best_s1 --max-windows 2000000 \
  >> lens/runs/GPU-BEST.txt 2>&1
