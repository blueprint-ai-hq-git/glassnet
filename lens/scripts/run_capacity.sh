#!/bin/bash
# FIXING THE MEMORISATION GAP (measured 2026-10-29: at 20k windows train 38-39% against held-out 10-12%,
# gap +26; at 2M windows the gap collapses to +3). Same data, same method, same steps - only the model's
# WIDTH changes, because width is the cheapest cause of raw memorisation: a d=256 model has 3.9M knobs and
# 20k windows cannot fill them, so it stores instead of generalising.
# Reference points to beat, measured: 20k windows at d=256 -> held-out 10.08/11.96%, gap +26/+29.
# PREDICTIONS, before any arm:
#  1. held-out accuracy RISES as width shrinks (64 < 128 < 256 parameters) while the gap FALLS. That is the
#     signature of "too much capacity for the data".
#  2. if held-out instead FALLS as width shrinks, we are underfitting and capacity is not the lever - the
#     bottleneck is the concept layer, and that is the next single change.
#  3. a d=32 arm is included as the control for "just make it tiny": if d=32 also wins on held-out, then the
#     task is being solved by capacity alone and the concept layer is doing nothing - that would be a finding
#     about the architecture, not a success.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { D=$1; SEED=$2
  echo "===== d $D windows 20000 seed $SEED  $(date '+%H:%M')" >> lens/runs/CAPACITY.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d $D --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 \
    --out "cap_d${D}_s$SEED" --method ours --opt-lr 1.0 >> lens/runs/CAPACITY.txt 2>&1
  echo "  d $D seed $SEED exit=$?" >> lens/runs/CAPACITY.txt
}
for D in 32 64 128; do run $D 0; run $D 1; done
# the 20k width reference at d=256 is already measured: 10.08 / 11.96 held-out, gap +26 / +29
echo "=== CAPACITY DONE $(date '+%F %H:%M')" >> lens/runs/CAPACITY.txt
