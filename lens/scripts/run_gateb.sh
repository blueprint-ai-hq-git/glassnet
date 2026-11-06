#!/bin/bash
# GATE B (2026-11-02). Everything about shape was tested at 20k windows, where NOTHING learns - so those results
# cannot answer whether attention helps. This runs the same comparison where the model does learn (2M windows),
# at batch 1024 (measured 44% faster than 256), and records the talk-quality number too, not just accuracy.
# Reference: no-attention at 2M windows measured 26.97 / 27.28 held-out (two seeds).
# PREDICTIONS, before any arm:
#  1. if token order is what unlocks meaning, attention beats 27.1% clearly at this data size.
#  2. if attention still loses here, it is not the lever at any data size we can afford, and the
#     conclusion is architectural (compute path), not a knob - that closes the question for good.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run() { NAME=$1; shift
  echo "===== $NAME : $*  $(date '+%H:%M')" >> lens/runs/GATEB.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 2000000 \
    --out "gb_$NAME" "$@" >> lens/runs/GATEB.txt 2>&1
  echo "  $NAME exit=$?" >> lens/runs/GATEB.txt
}
run attn --attention
run noattn
echo "=== GATEB DONE $(date '+%F %H:%M')" >> lens/runs/GATEB.txt
