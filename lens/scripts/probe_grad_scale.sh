#!/bin/bash
# probe_grad_scale.sh -- CALIBRATION ONLY, no training, no accuracy claim.
#
# Runs ONE batch's forward/backward through the REAL data path, mask and loss
# (train_chat.py --probe-grads) and appends the per-tensor gradient norms to
# lens/runs/OPT-GRAD-SCALE.txt. Exists because our rule (lens/opt.py RelativeGrad)
# normalises the WHOLE tensor, so its per-entry movement is opt_lr/sqrt(n) and the
# step-size grid for PLAN item (d2b) has to come from a measurement on this net
# rather than from the toy numeric-path numbers.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
OUT=lens/runs/OPT-GRAD-SCALE.txt
{
  echo "=== PROBE START $(date '+%F %H:%M:%S')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --train-turn-end \
    --max-windows 150000 --probe-grads --out _probe_grads
  echo "PROBE EXIT=$?"
  echo "=== PROBE DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
