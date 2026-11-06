#!/bin/bash
# AFTER THE CAPACITY RESULT (2026-10-30): at 20k windows held-out stayed 9.0-12.0% at EVERY width
# (d=32 9.02, d=64 9.97, d=128 10.06, d=256 11.02 mean) while the train-held-out gap fell from +26 to +10.
# So width controls MEMORISATION, not learning. The wall at 20k is something else. Two candidates remain:
# (a) DATA limit - 20k windows (640k tokens) simply is not enough, or (b) OPTIMISATION limit - the model is
# seen only 3 times over that data and has not converged. This arm tells those apart with one change: passes.
# PREDICTIONS, before any arm:
#  1. if it is optimisation-bound, held-out at 12 epochs rises clearly above the 3-epoch 11.02% mean while the
#     train accuracy also rises. Then the wall was passes, not data, and "few examples" is still alive.
#  2. if it is data-bound, held-out stays at ~10-12% and only TRAIN accuracy climbs (memorising harder).
#     That would mean the model needs data volume and the architecture must change instead.
#  3. either way this is the honest fork; no redefining the bar after seeing it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { EP=$1; SEED=$2
  echo "===== epochs $EP windows 20000 d256 seed $SEED  $(date '+%H:%M')" >> lens/runs/PASSES.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs $EP --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 20000 --gate-tol 0.05 \
    --out "ps_e${EP}_s$SEED" --method ours --opt-lr 1.0 >> lens/runs/PASSES.txt 2>&1
  echo "  epochs $EP seed $SEED exit=$?" >> lens/runs/PASSES.txt
}
run 12 0
run 12 1
run 30 0
echo "=== PASSES DONE $(date '+%F %H:%M')" >> lens/runs/PASSES.txt
