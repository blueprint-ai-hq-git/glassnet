#!/bin/bash
# STEP 1 OF YAHYA'S SEQUENCE: TRAIN THE REAL MODEL WITH **OUR** METHOD, NOT THE OLD LOOP.
#
# The mistake this fixes: our method existed (lens/unified.py + lens/opt.py) but --opt defaulted to sgd and
# --gates defaulted to off, so every run reported so far trained on the OLD loop with our method switched off.
# Settings measured that way are settings for the old loop. This is the first run where the model is trained
# BY OUR METHOD: our step rule (relgrad, lr from a measurement not a taste), our memorisation gate and our
# agreement gate, all on, one switch (--method ours).
#
# REFERENCE, same data and size on the OLD loop (lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt, by-position mask):
#   train 38.57% / held-out 35.26% / ppl 40.42 at epoch 3 - a train-minus-held-out gap of 3.31 points.
#
# PREDICTIONS, written BEFORE the run:
#  1. THE GATE MUST ACT. Our memorisation gate decides per step whether to accept or refuse an update. If the
#     run reports ~0% refused, the gate is not really acting and that is a finding against it, not a clean run.
#  2. THE GAP SHOULD NARROW. If our method does what it claims, the train-minus-held-out gap comes in BELOW the
#     old loop's 3.31 points. That is the concrete claim being tested.
#  3. HELD-OUT SHOULD NOT COLLAPSE. If held-out falls well under 35%, our method is buying the narrower gap
#     with accuracy and that has to be said plainly rather than reported as a win on the gap alone.
#  4. THE TALK TEST DECIDES, as always.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== OUR METHOD, FIRST REAL RUN (d=256, full 16.9M corpus, 3 epochs) $(date '+%F %H:%M')"
  echo "reference (old loop, same data/size): train 38.57% held-out 35.26% ppl 40.42, gap 3.31 points"
  echo "predictions: gate must refuse >0% of steps; gap below 3.31; held-out not far under 35; talk test decides"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --method ours --out ours_d256_full_3
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/OURS-METHOD-RUN1.txt 2>&1
