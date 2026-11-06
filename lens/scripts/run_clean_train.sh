#!/bin/bash
# Train on the CLEAN control-token corpus, so every accuracy figure comes from a val that is disjoint from
# train at the training window (VERDICT: CLEAN, 0 overlapping windows, confirmed by exact bytes).
#
# The previous run (gn_d256_full_24) trained on prepared_mix, whose val had 48,615 windows appearing
# verbatim in train -- so its 32.72% is partly a memorisation score and is labelled contaminated. This run
# exists to replace that number with an honest one.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/CLEAN-TRAIN.txt
{
  echo "=== CLEAN CORPUS TRAIN $(date '+%F %H:%M')"
  echo "corpus data/chat/prepared_mix_ctrl_honest: 294,225,090 tokens, val disjoint from train (VERDICT CLEAN)"
  echo "recipe: d=256, window 32, 24 epochs, batch 256, lr 0.4, assistant-masked, turn-end targets"
  echo "prediction: masked accuracy will come out LOWER than the contaminated 32.72%, and that lower number"
  echo "is the real one. A drop here is the measurement working, not the model getting worse."
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 1500000 --out cl_d256_clean_24
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
