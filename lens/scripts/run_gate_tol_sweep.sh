#!/bin/bash
# FIX ATTEMPT: our method's gate refused 78-84% of updates and cost ~20 points of held-out accuracy
# (12.68% against 35.26% for the old loop on the same corpus and size). The refusal decision is governed by
# --gate-tol: the tolerance at which a candidate update is allowed through. Default is 0.0 (refuse unless the
# held-out side improves). These four arms raise it.
# PREDICTIONS, before any arm runs:
#  1. raising the tolerance LOWERS the refusal rate. If the refusal rate stays ~84% at tol 0.10, the tolerance
#     is not the knob and the gate is refusing for another reason - that is a finding, not a failure.
#  2. if refusals were the cost, held-out rises from 12.7% toward the old loop's 35%. If held-out stays near
#     12-15% while refusals drop, refusals were NOT the cause and the next suspect is our step rule's lr.
#  3. whichever arm is best gets 3 seeds. Nothing is called consistent on one seed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
for TOL in 0.0 0.01 0.05 0.10; do
  echo "===== gate tolerance $TOL  $(date '+%H:%M')" >> lens/runs/GATE-TOL-SWEEP.txt
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 2 --batch 1024 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --direct --direct-mode tied --max-windows 1000000 --method ours --gate-tol $TOL \
    --out gtol_$TOL >> lens/runs/GATE-TOL-SWEEP.txt 2>&1
  echo "  tol $TOL exit=$? $(grep -oE 'refused [0-9]+/[0-9]+' lens/runs/GATE-TOL-SWEEP.txt | tail -1)" >> lens/runs/GATE-TOL-SWEEP.txt
done
echo "=== SWEEP DONE $(date '+%F %H:%M')" >> lens/runs/GATE-TOL-SWEEP.txt
