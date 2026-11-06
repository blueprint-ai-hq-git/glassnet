#!/bin/bash
# Start real GPU training immediately on the valid full merged corpus (294,057,312 tokens).
# The control-token rebuild is CPU-only data preparation and runs alongside; the GPU must not sit idle
# while it finishes. Standard chat format is not required for the talk objective, so this trains now.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/GPU-NOW.txt
{
  echo "=== GPU NOW $(date '+%F %H:%M')"
  echo "corpus data/chat/prepared_mix: 294,057,312 tokens (the VALID full merge; not the truncated ctrl copy)"
  echo "recipe: d=256, window 32, 24 epochs, batch 256, lr 0.4, assistant-masked, relgrad-compatible settings"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 1500000 --out gn_d256_full_24
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
