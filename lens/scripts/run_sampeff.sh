#!/bin/bash
# SAMPLE EFFICIENCY VIA THE LEARNING RULE (2026-10-31). Eliminated so far at 20k windows: width (9-12% at every
# d), passes (3->30 epochs made it worse), attention (worse), tokens per example (worse), concept supervision
# (a tie). All shape. These arms change what the model is asked to LEARN:
#   A --mtp 2      : also predict the token TWO steps ahead from the same concept vector (published benefit is
#                    largest at small data, which is precisely where we are stuck)
#   B --drop-unk   : stop paying loss for target positions whose answer is <unk> (unanswerable by construction).
#                    Those positions are pure noise in the loss, and noise costs sample efficiency.
# Reference, measured at 20k windows d256 3 epochs: held-out 10.08 / 11.96, mean 11.02%.
# PREDICTIONS, before any arm:
#  1. if the wall is the learning signal rather than the data, A and/or B beat 11.02% clearly.
#  2. if both land at ~11%, then no learning-rule change available in this trainer moves it, and the honest
#     conclusion is that this architecture needs the data volume - which changes the goal, not the code.
#  3. any winner gets a third seed before it is called a win.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; SEED=$2; shift 2
  echo "===== $NAME seed $SEED : $*  $(date '+%H:%M')" >> lens/runs/SAMPEFF.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 \
    --out "se_${NAME}_s$SEED" --method ours --opt-lr 1.0 "$@" >> lens/runs/SAMPEFF.txt 2>&1
  echo "  $NAME seed $SEED exit=$?" >> lens/runs/SAMPEFF.txt
}
run mtp2 0 --mtp 2
run mtp2 1 --mtp 2
run dropunk 0 --drop-unk
run dropunk 1 --drop-unk
echo "=== SAMPEFF DONE $(date '+%F %H:%M')" >> lens/runs/SAMPEFF.txt
