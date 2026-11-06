#!/bin/bash
# ISOLATING A/B: identical tiny config, 8 epochs, one corpus per arm.
# Question: is the merged corpus itself untrainable, or was M0's collapse specific to that run?
# Each arm prints its trajectory. A healthy arm shows val_acc climbing above the constant baseline
# (4.2%); the merged arm collapsing while the splitfix arm climbs would localise the fault to the
# corpus + trainer interaction rather than the model, the resume, or the eval.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
exec >> /tmp/ab_corpus.log 2>&1
echo "=== A/B START $(date '+%H:%M')"
for CORPUS in prepared_ext prepared_splitfix; do
  echo "--- arm: $CORPUS"
  CHAT_DATA_DIR="$PWD/data/chat/$CORPUS" nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 128 --window 32 --epochs 8 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --seed 0 --device gpu --assistant-only --max-windows 300000 \
    --out ab_$CORPUS 2>&1 | grep -E "windows|cached|ep |window cap|subsample"
done
echo "=== A/B DONE $(date '+%H:%M')"