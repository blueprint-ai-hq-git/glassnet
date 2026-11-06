#!/bin/bash
# THE REAL BRAKE, FROM THE MEASUREMENT: at tol 0.01+ the gate refused 0 of 1,340 steps and train accuracy was
# still only 13.48%, against ~38-41% for the old loop at this size. So the gate was NOT the cost. What is left:
# (a) our step rule's step size (opt-lr), and (b) the agreement gate, which still refused 31.4% of steps.
# This sweep tests exactly those two, plus a matched plain-SGD arm as the reference at identical settings.
# PREDICTIONS, before any arm:
#  1. if the step size is the brake, the higher opt-lr arms rise well above 13.5% and toward the SGD reference.
#  2. if the agreement gate is the brake, the gate-off arm rises while the others stay flat.
#  3. the arm closest to the SGD reference tells us how much of the gap is our method and how much is the recipe.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; shift
  echo "===== $NAME : $*  $(date '+%H:%M')" >> lens/runs/LR-AGREE-SWEEP.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 1000000 --gate-tol 0.05 \
    --out "la_$NAME" "$@" >> lens/runs/LR-AGREE-SWEEP.txt 2>&1
  echo "  $NAME exit=$?" >> lens/runs/LR-AGREE-SWEEP.txt
}
run ours_lr005  --method ours --opt-lr 0.05
run ours_lr015  --method ours --opt-lr 0.15
run ours_lr050  --method ours --opt-lr 0.50
run ours_noagree --method ours --opt-lr 0.15 --gates gap
run sgd_ref     --method standard
echo "=== SWEEP DONE $(date '+%F %H:%M')" >> lens/runs/LR-AGREE-SWEEP.txt
