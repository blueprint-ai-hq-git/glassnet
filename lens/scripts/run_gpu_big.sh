#!/bin/bash
# Batch 4096 measured 5.46M tok/s vs 3.42M at 1024, so the GPU stops idling between small chunks.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 4096 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --direct --direct-mode tied --out gpu_big_b4096 --max-windows 2000000 \
  >> lens/runs/GPU-BIG.txt 2>&1
