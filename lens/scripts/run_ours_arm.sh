#!/bin/bash
# ONE ARM OF OUR METHOD'S SETTINGS SWEEP. d=256, 2,000,000 windows, 2 epochs: small enough that each arm finishes
# in about 20 minutes at 2 cores, big enough to separate a real effect from noise. Same seed, same corpus.
# PREDICTIONS, written BEFORE any arm runs:
#  1. arm G (the old loop) should land near the 35.26% reference, which also checks that shrinking to 2M windows
#     did not itself change the picture.
#  2. if the refusals are what cost our method its accuracy, then loosening the tolerance (B, C) raises held-out
#     and LOWERS the refusal rate. If held-out stays near 12.68% while refusals drop, the refusals were NOT the
#     cause and the cause is elsewhere (likely our step rule's lr or the gate measuring on a probe too small).
#  3. the arm with the best held-out wins; its setting is then re-run at 3 SEEDS before being called consistent.
ARM="$1"; shift
OUT="ours_arm_${ARM}"
{
  echo "---------------------------------------------------------------"
  echo "ARM ${ARM}  flags: $*   $(date '+%F %H:%M')"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 2000000 --out "$OUT" "$@"
  echo "exit=$?"
} 
>> lens/runs/OURS-SWEEP.txt 2>&1
