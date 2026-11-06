#!/bin/bash
# QUEUED EXPERIMENT: window length 16 -> 24, on top of whatever vocab wins.
#
# WHY THIS IS THE NEXT ONE-AT-A-TIME CHANGE: the earlier sweep (run_best.sh header) measured that
# window 16 beat window 8 on accuracy (24.69% vs baseline) while vocab 3500 won on ppl. Context
# length is a DIFFERENT resource from vocabulary width, so it is a legitimate second lever -- but it
# must be tested SEPARATELY, after v12k's verdict, never stacked blindly.
#
# PRE-REGISTERED (written BEFORE the run):
#   w1  val_ppl at W=24 beats the W=16 run at the SAME vocab by > 3%   (longer context helps)
#   w2  bits/token improves by > 0.05                                  (real, not noise)
#   w3  if w1 fails, longer context does NOT help this model at this data size -- report it as a
#       failure and stop adding context (do not keep stacking changes hoping something moves).
#
# This script WAITS for any running chat training to finish first (same guard the project uses),
# so it can be launched now and will not fight for the GPU.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

VOCAB="${W24_VOCAB:-12000}"
echo "waiting for the GPU to be free (v12k must finish first)..."
while pgrep -f "train_chat.py" > /dev/null; do
  sleep 20
done
echo "GPU free; starting the window-24 run at vocab $VOCAB"

CD_VOCAB=$VOCAB TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 24 --epochs 24 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --device gpu --out "w24_v${VOCAB}" \
  --max-windows 1500000 2>&1 | grep -E "^  ep |final VAL|saved|Error|Traceback" | tail -30
