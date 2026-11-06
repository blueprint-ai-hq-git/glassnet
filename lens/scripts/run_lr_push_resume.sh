#!/bin/bash
# RESUMED after the accidental power cut (2026-10-28). The lr-push sweep was killed mid-arm. What it had
# established BEFORE the cut: our method at opt-lr 0.50 = 22.50% held-out val, at 1.0 = 23.22%, against
# plain SGD's 18.93% at identical settings. Its remaining arms are the 3-SEED PROOF, which is the only thing
# that can turn one lucky seed into a settled setting.
# PREDICTIONS, before any arm: the lr 0.50 mean over seeds 0/1/2 lands within ~2 points of 22.5% and stays
# above the SGD mean; if the spread is wider than 4 points the setting is NOT settled and gets re-swept.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; SEED=$2; shift 2
  echo "===== $NAME seed $SEED : $*  $(date '+%H:%M')" >> lens/runs/LR-PUSH.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 1000000 --gate-tol 0.05 \
    --out "lpr_${NAME}_s$SEED" "$@" >> lens/runs/LR-PUSH.txt 2>&1
  echo "  $NAME seed $SEED exit=$?" >> lens/runs/LR-PUSH.txt
}
run ours_lr050_s0 0 --method ours --opt-lr 0.50
run ours_lr050_s2 2 --method ours --opt-lr 0.50
run ours_lr100_s0 0 --method ours --opt-lr 1.0
run ours_lr100_s1 1 --method ours --opt-lr 1.0
run sgd_ref_s0 0 --method standard
run sgd_ref_s1 1 --method standard
echo "=== DONE $(date '+%F %H:%M')" >> lens/runs/LR-PUSH.txt
