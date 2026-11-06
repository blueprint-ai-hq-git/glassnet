#!/bin/bash
# Where does the honest val accuracy actually PEAK? Measured, because both finished runs show the val turning
# DOWN while the train accuracy climbs: d=256 peaked around 27.5% near ep22-24, and d=512 is already falling at
# ep34 (24.28%). So "train longer" is not a setting, it is a way to overfit. This sweeps training length at
# d=256 on the clean corpus with everything else fixed, and reports the epoch where the honest val is best.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/EPOCH-SWEEP.txt
{
  echo "=== EPOCH SWEEP (d=256, clean corpus) $(date '+%F %H:%M')"
  echo "fixed: window 32, batch 256, lr 0.4 decay 0.5, assistant-masked, seed 0, 1.5M windows"
  echo "question: which training length gives the best HONEST masked val accuracy, and by how much does"
  echo "overfitting cost at 48 epochs? (d=512 ep34 already fell to 24.28% from the 27.24% peak)"
  for EP in 6 12 24; do
    echo "--- epochs=$EP"
    CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
      lens/scripts/train_chat.py --d 256 --window 32 --epochs $EP --batch 256 --lr 0.4 --lr-decay 0.5 \
      --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
      --max-windows 1500000 --out sweepep$EP 2>&1 | grep -E "^  ep "
    echo "exit=$?"
  done
  echo "=== SWEEP DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
