#!/bin/bash
# THE ARCHITECTURE QUESTION, first cut with flags that exist today. The concept layer squeezes 8192 numbers
# into 36 named neurons before answering, and the model is trained to make those 36 match human labels
# (--lam-concept 0.3, weight of that label-matching loss). If that supervision forces the layer to encode
# LABELS instead of what the task needs, it costs accuracy - and that is testable by changing ONE weight.
# Reference, measured at 200k windows d256 3 epochs: held-out 17.90 / 20.63% (mean 19.27).
# PREDICTIONS, before any arm:
#  1. if concept supervision costs quality, lam 0.0 beats 19.27% clearly.
#  2. if it helps the model, lam 1.0 beats lam 0.3 (more supervision, better features).
#  3. whichever wins gets a third seed. A tie means the funnel is NOT the wall and the next change is the
#     readout width, not the supervision.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { LAM=$1; SEED=$2
  echo "===== lam-concept $LAM seed $SEED  $(date '+%H:%M')" >> lens/runs/LAMCONCEPT.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept $LAM --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 200000 --gate-tol 0.05 \
    --out "lc_lam${LAM}_s$SEED" --method ours --opt-lr 1.0 >> lens/runs/LAMCONCEPT.txt 2>&1
  echo "  lam $LAM seed $SEED exit=$?" >> lens/runs/LAMCONCEPT.txt
}
run 0.0 0
run 0.0 1
run 1.0 0
run 1.0 1
echo "=== LAMCONCEPT DONE $(date '+%F %H:%M')" >> lens/runs/LAMCONCEPT.txt
