#!/bin/bash
# Resume the full-corpus run from its newest checkpoint (epoch 2 of 24) instead of restarting it.
# Corpus: data/chat/prepared_mix -- 294,057,312 tokens of talk + maths + knowledge + code.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/GPU-NOW.txt
CK=$(ls -t lens/runs/chat/gn_d256_full_24_ep*.npz 2>/dev/null | head -1)
EP=$(basename "$CK" .npz 2>/dev/null | sed 's/.*_ep//')
{
  echo "=== RESUME FULL CORPUS $(date '+%F %H:%M') from ${CK:-scratch} (epoch ${EP:-0} of 24)"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 1500000 --out gn_d256_full_24 \
    $([ -n "$CK" ] && echo "--resume $CK --resume-ep $EP")
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
