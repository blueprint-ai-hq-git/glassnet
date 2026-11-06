#!/bin/bash
# OUR METHOD, sized to complete. The first attempt (run_ours_first.sh, 16.9M windows) was SIGTERM'd at 85 minutes
# with NO epoch finished, so it produced no comparable number at all. Cause: the job was torn down with the
# scope that launched it (exit 143). The launcher is now hardened (KillMode=process, infinite stop timeout).
#
# WHAT THE DEAD RUN DID TELL US, and it is a real result: our memorisation gate REFUSED 3,822 of 4,925 steps
# (78%), and the agreement gate refused 2,500 of 7,425 (34%). Our method is FAR more conservative than the old
# loop, which is exactly why it needs its own settings instead of the old loop's - the point Yahya made.
#
# PREDICTIONS, written BEFORE this run:
#  1. an epoch will COMPLETE this time (that is the minimum bar the last run failed).
#  2. the gate refusal rate stays high (well above 0%). If it drops near zero the gate is not acting.
#  3. held-out will be LOW compared with the old loop's 35.26% on the same corpus, because refusing 78% of
#     updates must slow learning. If held-out is comparable despite that, the refusals are doing something
#     useful rather than just costing progress.
#  4. the talk test decides, as always.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== OUR METHOD, SIZED TO COMPLETE (d=256, 3M windows, 2 epochs) $(date '+%F %H:%M')"
  echo "prior attempt: 16.9M windows SIGTERM'd at 85 min, no epoch finished; gate refused 3822/4925 steps (78%)"
  echo "predictions: epoch completes; refusal rate stays well above 0%; held-out likely below 35.26%; talk decides"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 3000000 --method ours --out ours_d256_3M_2ep
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/OURS-METHOD-RUN2.txt 2>&1
