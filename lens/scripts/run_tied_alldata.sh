#!/bin/bash
# CRITICAL PATH 1 OF 2: THE UNDERTRAINING TEST.
# The tied run could finally FIT (train accuracy 24.8% -> 44.14%) but it overfit by epoch 4, and the trainer
# itself printed "tokens per parameter: 0.36 (UNDER-TRAINED)". Every run so far used 1,500,000 of the
# ~16,900,000 available windows - 9% of the corpus - so the model may simply have been shown too little text.
# This run feeds it 6,000,000 windows (3.5x the usual 1.5M). The FULL 16.9M was tried first and had to be
# stopped: RSS reached 9.0 GB and was still climbing on a 14 GB machine, the same shape as the 15:48 OOM kill.
# The full-corpus run needs the chunked window sampler before it is safe; that is noted in STATE.md., same architecture, same seed, best-val checkpointing on.
#
# PREDICTIONS, written BEFORE the run:
#  1. if the undertraining explanation is right, held-out should beat the tied run's peak 29.71% and the talk
#     test should improve. 
#  2. if held-out stays in the 27-30% band, then data volume was NOT the binding constraint and the next test
#     is capacity (d=512, already queued as run 2) - not another variation of this one.
#  3. the gap between train and held-out is expected to stay wide; that is the memorisation question, which is
#     measured here and attacked separately, not hidden by the gate.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS (~16.9M windows) d=256 3 epochs $(date '+%F %H:%M')"
  echo "reference: tied/1.5M-windows peaked 29.71% (ep4), ended 27.37%, train acc 44.14%, talk still salad"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 6000000 --out ct_d256_alldata_3
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/TIED-ALLDATA.txt 2>&1
