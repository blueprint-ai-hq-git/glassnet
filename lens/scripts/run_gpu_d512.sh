#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
# d is pinned to 512 exactly once: the old file passed --d 256 early and --d 512 later,
# so the run only got 512 because argparse keeps the last value. Fixed 2026-11-06.
CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 512 --window 32 --epochs 3 --batch 1024 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --direct --direct-mode tied --max-windows 2000000 --seed 0 --out gpu_d512_s0 --batch 1024 \
  >> lens/runs/GPU-QUEUE-RUNS.txt 2>&1
