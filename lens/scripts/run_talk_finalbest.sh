#!/bin/bash
# THE TALK TEST ON THE BEST CHECKPOINT OF THE FULL-CORPUS MASK-FIXED ARM (the artifact that would ship).
# Predictions P84a-f written BEFORE this ran: lens/runs/TALK-FINALBEST-PREDICTIONS-P84.txt
# CPU work, nice 19 / ionice -c3, its own output files with their own tag; nothing stored is overwritten.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TALK TEST, FINAL BEST (P84a-f) $(date '+%F %H:%M') ==="
  echo "checkpoint: lens/runs/chat/ct_d256_full169_fix3_best.npz (ep3, by-position held-out 35.26%)"
  echo "the ep1 result on this same arm for comparison: lens/runs/TALK-FAIR-maskfix-ep1.txt"
  TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-finalbest-d256.txt" \
  TALK_FAIR_LOGGED=0.3526 TALK_FAIR_EP="ep3 (FINAL BEST of ct_d256_full169_fix3)" \
  TALK_FAIR_PREDICTIONS="lens/runs/TALK-FINALBEST-PREDICTIONS-P84.txt" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TALK-FINALBEST-RUN.txt 2>&1
