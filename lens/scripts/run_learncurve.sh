#!/bin/bash
# SAMPLE EFFICIENCY - Yahya's core idea, tested for the first time: a small model must LEARN from few
# examples, not memorise a huge corpus. Design: four training-set sizes, same model, same method (our own,
# opt-lr 1.0 - the setting found on our method), same number of passes for the small ones so they get a fair
# chance, 2 seeds each. The number that answers the question is the GAP between train accuracy and held-out
# accuracy: a small gap at 2k windows = it generalises; a huge gap that only closes with 2M windows = memorising.
# PREDICTIONS, written before any arm runs:
#  1. held-out accuracy rises with training-set size (normal). The INTERESTING question is the shape.
#  2. if the 200k result is within ~3 points of the 2M result, the model does NOT need the whole corpus - that
#     supports "learns rules". If 2M is far ahead, it is consuming data, which is the memorisation answer.
#  3. the 2k arm is expected to overfit badly: its train-vs-heldout gap should be the largest. If 2k is already
#     close to 200k, the probes are broken and the numbers are meaningless - that would be a bug, not a win.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { SZ=$1; SEED=$2
  echo "===== windows $SZ seed $SEED  $(date '+%H:%M')" >> lens/runs/LEARNCURVE.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows $SZ --gate-tol 0.05 \
    --out "lc_w${SZ}_s$SEED" --method ours --opt-lr 1.0 >> lens/runs/LEARNCURVE.txt 2>&1
  echo "  windows $SZ seed $SEED exit=$?" >> lens/runs/LEARNCURVE.txt
}
for SZ in 2000 20000 200000 2000000; do run $SZ 0; run $SZ 1; done
echo "=== LEARNCURVE DONE $(date '+%F %H:%M')" >> lens/runs/LEARNCURVE.txt
