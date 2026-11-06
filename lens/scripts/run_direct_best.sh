#!/bin/bash
# THE RUN THAT ANSWERS THE REAL QUESTION: does the direct path fix the word salad in the CHAT model?
#
# Same setting as the current best honest baseline so the comparison is clean:
#   d=256, window 32, 12 epochs, batch 256, lr 0.4 decay 0.5, assistant-masked, clean corpus, seed 0.
# Baseline to beat: no-attention, no-direct d=256/12ep = 28.30% final, peak 28.52% at ep9 (clean val, n=44,418).
#
# PREDICTION, written BEFORE the run (required rule):
#   1. the honest masked val will NOT drop below 28% - the direct path adds capacity without removing the
#      concept path, so the classifier should hold or improve. If it drops, the direct path is interfering and
#      that is a real finding against it.
#   2. the training loss will go LOWER than 3.48 (the no-direct 12-epoch run ended at 3.48) because the model
#      can finally fit token-level detail. A clearly lower loss with a flat val would mean it is fitting noise.
#   3. the talk test is the honest verdict, not the loss: if the text is still salad, the direct path alone was
#      not enough and the state/label work (item B) is required. Recorded either way.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== DIRECT PATH, BEST SETTING (d=256, 12 ep, clean corpus) $(date '+%F %H:%M')"
  echo "baseline: no-attention no-direct d=256/12ep = 28.30% final / 28.52% peak, train loss ended 3.48"
  echo "prediction: val holds >=28%, train loss goes lower than 3.48, talk test decides"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-rank 16 --max-windows 1500000 --out cl_d256_direct_12
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/DIRECT-BEST.txt 2>&1
