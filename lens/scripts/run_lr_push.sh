#!/bin/bash
# RESULT SO FAR: our method at opt-lr 0.50 gave 22.50% val against plain SGD's 18.93% at identical settings -
# our method WINS once its step size is right, and accuracy rose monotonically with lr (15.63 -> 19.95 -> 22.50).
# The agreement gate was NOT the brake (gate-off arm identical to gate-on). So the next question is only:
# where does the step size stop helping, and does the win survive 3 seeds.
# PREDICTIONS, before any arm:
#  1. lr 1.0 and 2.0 either keep rising or turn DOWN (instability); if val ppl explodes, that is the ceiling.
#  2. the winner then re-runs at seeds 0/1/2; the win is only real if it holds on all three.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; SEED=$2; shift 2
  echo "===== $NAME seed $SEED : $*  $(date '+%H:%M')" >> lens/runs/LR-PUSH.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 1000000 --gate-tol 0.05 \
    --out "lp_${NAME}_s$SEED" "$@" >> lens/runs/LR-PUSH.txt 2>&1
  echo "  $NAME seed $SEED exit=$?" >> lens/runs/LR-PUSH.txt
}
run ours_lr100 0 --method ours --opt-lr 1.0
run ours_lr200 0 --method ours --opt-lr 2.0
run ours_lr050_s1 1 --method ours --opt-lr 0.50
run ours_lr050_s2 2 --method ours --opt-lr 0.50
run sgd_ref_s1 1 --method standard
run sgd_ref_s2 2 --method standard
echo "=== DONE $(date '+%F %H:%M')" >> lens/runs/LR-PUSH.txt
